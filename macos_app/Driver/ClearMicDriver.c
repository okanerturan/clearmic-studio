// ClearMic.driver — Native Apple Silicon CoreAudio HAL AudioServerPlugIn.
// Creates "ClearMic Virtual Mic" (48 kHz, 2-ch float32 loopback virtual mic).
// ClearMic Studio writes cleaned 48 kHz voice to its output side; Zoom, Google Meet,
// Discord, and System Settings read it from its input side as a standard microphone.
// Output cannot be auto-selected as default system speakers (DeviceCanBeDefaultDevice=0),
// so it never hijacks your AirPods or speakers.

#include <CoreAudio/AudioServerPlugIn.h>
#include <CoreFoundation/CoreFoundation.h>
#include <dispatch/dispatch.h>
#include <mach/mach_time.h>
#include <os/lock.h>
#include <pthread.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

// Factory UUID (must match Info.plist CFPlugInFactories)
// 7F5A2B9C-4D31-4E8A-9C12-6E3B8A1F4D20
#define kClearMicFactoryUUID CFUUIDGetConstantUUIDWithBytes(NULL, \
    0x7F, 0x5A, 0x2B, 0x9C, 0x4D, 0x31, 0x4E, 0x8A, \
    0x9C, 0x12, 0x6E, 0x3B, 0x8A, 0x1F, 0x4D, 0x20)

#define kDevice_Name         "ClearMic Virtual Mic"
#define kDevice_Manufacturer "ClearMic Studio"
#define kDevice_UID          "com.clearmic.virtualmic.device"
#define kDevice_ModelUID     "com.clearmic.virtualmic.model"

enum {
    kObjectID_PlugIn        = kAudioObjectPlugInObject, // 1
    kObjectID_Device        = 2,
    kObjectID_Stream_Input  = 3,
    kObjectID_Stream_Output = 4,
};

#define kSampleRate         48000.0
#define kNumChannels        2
#define kRingFrames         16384  // Power of 2 (~341 ms @ 48 kHz)
#define kRingMask           (kRingFrames - 1)
#define kZeroStampPeriod    16384  // Frames per zero-timestamp cycle

// Global driver state
static AudioServerPlugInHostRef gHost = NULL;
static pthread_mutex_t          gStateMutex = PTHREAD_MUTEX_INITIALIZER;
static UInt32                   gIOClientCount = 0;
static UInt32                   gOutputActive = 0;
static Float64                  gHostTicksPerFrame = 0.0;
static UInt64                   gNumberTimeStamps = 0;
static UInt64                   gAnchorHostTime = 0;
static Float64                  gAnchorSampleTime = 0.0;

// Lock-protected FIFO loopback ring buffer (immune to client buffer-size / sample-time offsets)
static os_unfair_lock           gRingLock = OS_UNFAIR_LOCK_INIT;
static float                    gRingBuf[kRingFrames * kNumChannels];
static UInt32                   gRingRead = 0;
static UInt32                   gRingWrite = 0;
static UInt32                   gRingCount = 0;

// Forward declarations
static HRESULT  CM_QueryInterface(void* inDriver, REFIID inUUID, LPVOID* outInterface);
static ULONG    CM_AddRef(void* inDriver);
static ULONG    CM_Release(void* inDriver);
static OSStatus CM_Initialize(AudioServerPlugInDriverRef inDriver, AudioServerPlugInHostRef inHost);
static OSStatus CM_CreateDevice(AudioServerPlugInDriverRef inDriver, CFDictionaryRef inDescription, const AudioServerPlugInClientInfo* inClientInfo, AudioObjectID* outDeviceObjectID);
static OSStatus CM_DestroyDevice(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID);
static OSStatus CM_AddDeviceClient(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, const AudioServerPlugInClientInfo* inClientInfo);
static OSStatus CM_RemoveDeviceClient(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, const AudioServerPlugInClientInfo* inClientInfo);
static OSStatus CM_PerformDeviceConfigurationChange(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, UInt64 inChangeAction, void* inChangeInfo);
static OSStatus CM_AbortDeviceConfigurationChange(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, UInt64 inChangeAction, void* inChangeInfo);
static Boolean  CM_HasProperty(AudioServerPlugInDriverRef inDriver, AudioObjectID inObjectID, pid_t inClientProcessID, const AudioObjectPropertyAddress* inAddress);
static OSStatus CM_IsPropertySettable(AudioServerPlugInDriverRef inDriver, AudioObjectID inObjectID, pid_t inClientProcessID, const AudioObjectPropertyAddress* inAddress, Boolean* outIsSettable);
static OSStatus CM_GetPropertyDataSize(AudioServerPlugInDriverRef inDriver, AudioObjectID inObjectID, pid_t inClientProcessID, const AudioObjectPropertyAddress* inAddress, UInt32 inQualifierDataSize, const void* inQualifierData, UInt32* outDataSize);
static OSStatus CM_GetPropertyData(AudioServerPlugInDriverRef inDriver, AudioObjectID inObjectID, pid_t inClientProcessID, const AudioObjectPropertyAddress* inAddress, UInt32 inQualifierDataSize, const void* inQualifierData, UInt32 inDataSize, UInt32* outDataSize, void* outData);
static OSStatus CM_SetPropertyData(AudioServerPlugInDriverRef inDriver, AudioObjectID inObjectID, pid_t inClientProcessID, const AudioObjectPropertyAddress* inAddress, UInt32 inQualifierDataSize, const void* inQualifierData, UInt32 inDataSize, const void* inData);
static OSStatus CM_StartIO(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, UInt32 inClientID);
static OSStatus CM_StopIO(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, UInt32 inClientID);
static OSStatus CM_GetZeroTimeStamp(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, UInt32 inClientID, Float64* outSampleTime, UInt64* outHostTime, UInt64* outSeed);
static OSStatus CM_WillDoIOOperation(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, UInt32 inClientID, UInt32 inOperationID, Boolean* outWillDo, Boolean* outWillDoInPlace);
static OSStatus CM_BeginIOOperation(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, UInt32 inClientID, UInt32 inOperationID, UInt32 inIOBufferFrameSize, const AudioServerPlugInIOCycleInfo* inIOCycleInfo);
static OSStatus CM_DoIOOperation(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, AudioObjectID inStreamObjectID, UInt32 inClientID, UInt32 inOperationID, UInt32 inIOBufferFrameSize, const AudioServerPlugInIOCycleInfo* inIOCycleInfo, void* ioMainBuffer, void* ioSecondaryBuffer);
static OSStatus CM_EndIOOperation(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, UInt32 inClientID, UInt32 inOperationID, UInt32 inIOBufferFrameSize, const AudioServerPlugInIOCycleInfo* inIOCycleInfo);

static AudioServerPlugInDriverInterface gDriverInterface = {
    NULL,
    CM_QueryInterface,
    CM_AddRef,
    CM_Release,
    CM_Initialize,
    CM_CreateDevice,
    CM_DestroyDevice,
    CM_AddDeviceClient,
    CM_RemoveDeviceClient,
    CM_PerformDeviceConfigurationChange,
    CM_AbortDeviceConfigurationChange,
    CM_HasProperty,
    CM_IsPropertySettable,
    CM_GetPropertyDataSize,
    CM_GetPropertyData,
    CM_SetPropertyData,
    CM_StartIO,
    CM_StopIO,
    CM_GetZeroTimeStamp,
    CM_WillDoIOOperation,
    CM_BeginIOOperation,
    CM_DoIOOperation,
    CM_EndIOOperation,
};

static AudioServerPlugInDriverInterface* gDriverInterfacePtr = &gDriverInterface;
static AudioServerPlugInDriverRef        gDriverRef = &gDriverInterfacePtr;
static UInt32                            gRefCount = 1;

static AudioStreamBasicDescription MakeFormat(void) {
    AudioStreamBasicDescription fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.mSampleRate       = kSampleRate;
    fmt.mFormatID         = kAudioFormatLinearPCM;
    fmt.mFormatFlags      = kAudioFormatFlagIsFloat | kAudioFormatFlagsNativeEndian | kAudioFormatFlagIsPacked;
    fmt.mBytesPerPacket   = kNumChannels * sizeof(Float32);
    fmt.mFramesPerPacket  = 1;
    fmt.mBytesPerFrame    = kNumChannels * sizeof(Float32);
    fmt.mChannelsPerFrame = kNumChannels;
    fmt.mBitsPerChannel   = 32;
    return fmt;
}

// Entry point called by CoreAudio via CFPlugIn
void* ClearMicPlugIn_Create(CFAllocatorRef inAllocator, CFUUIDRef inRequestedTypeUUID) {
    (void)inAllocator;
    if (CFEqual(inRequestedTypeUUID, kAudioServerPlugInTypeUUID)) {
        return gDriverRef;
    }
    return NULL;
}

static HRESULT CM_QueryInterface(void* inDriver, REFIID inUUID, LPVOID* outInterface) {
    (void)inDriver;
    if (outInterface == NULL) return kAudioHardwareIllegalOperationError;
    CFUUIDRef req = CFUUIDCreateFromUUIDBytes(NULL, inUUID);
    if (CFEqual(req, IUnknownUUID) || CFEqual(req, kAudioServerPlugInDriverInterfaceUUID)) {
        CFRelease(req);
        gRefCount++;
        *outInterface = gDriverRef;
        return S_OK;
    }
    CFRelease(req);
    *outInterface = NULL;
    return E_NOINTERFACE;
}

static ULONG CM_AddRef(void* inDriver) {
    (void)inDriver;
    return ++gRefCount;
}

static ULONG CM_Release(void* inDriver) {
    (void)inDriver;
    if (gRefCount > 1) --gRefCount;
    return gRefCount;
}

static OSStatus CM_Initialize(AudioServerPlugInDriverRef inDriver, AudioServerPlugInHostRef inHost) {
    (void)inDriver;
    gHost = inHost;
    struct mach_timebase_info tb;
    mach_timebase_info(&tb);
    Float64 hostClockFreq = (Float64)tb.denom / (Float64)tb.numer * 1000000000.0;
    gHostTicksPerFrame = hostClockFreq / kSampleRate;
    memset(gRingBuf, 0, sizeof(gRingBuf));
    gRingRead = 0;
    gRingWrite = 0;
    gRingCount = 0;
    gOutputActive = 0;
    return noErr;
}

static OSStatus CM_CreateDevice(AudioServerPlugInDriverRef inDriver, CFDictionaryRef inDescription, const AudioServerPlugInClientInfo* inClientInfo, AudioObjectID* outDeviceObjectID) {
    (void)inDriver; (void)inDescription; (void)inClientInfo; (void)outDeviceObjectID;
    return kAudioHardwareUnsupportedOperationError;
}

static OSStatus CM_DestroyDevice(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID) {
    (void)inDriver; (void)inDeviceObjectID;
    return kAudioHardwareUnsupportedOperationError;
}

static OSStatus CM_AddDeviceClient(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, const AudioServerPlugInClientInfo* inClientInfo) {
    (void)inDriver; (void)inDeviceObjectID; (void)inClientInfo;
    return noErr;
}

static OSStatus CM_RemoveDeviceClient(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, const AudioServerPlugInClientInfo* inClientInfo) {
    (void)inDriver; (void)inDeviceObjectID; (void)inClientInfo;
    return noErr;
}

static OSStatus CM_PerformDeviceConfigurationChange(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, UInt64 inChangeAction, void* inChangeInfo) {
    (void)inDriver; (void)inDeviceObjectID; (void)inChangeAction; (void)inChangeInfo;
    return noErr;
}

static OSStatus CM_AbortDeviceConfigurationChange(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, UInt64 inChangeAction, void* inChangeInfo) {
    (void)inDriver; (void)inDeviceObjectID; (void)inChangeAction; (void)inChangeInfo;
    return noErr;
}

// MARK: - Property Helpers

static Boolean HasPlugInProperty(const AudioObjectPropertyAddress* a) {
    switch (a->mSelector) {
        case kAudioObjectPropertyBaseClass:
        case kAudioObjectPropertyClass:
        case kAudioObjectPropertyOwner:
        case kAudioObjectPropertyManufacturer:
        case kAudioObjectPropertyOwnedObjects:
        case kAudioPlugInPropertyDeviceList:
        case kAudioPlugInPropertyTranslateUIDToDevice:
        case kAudioPlugInPropertyResourceBundle:
            return true;
        default:
            return false;
    }
}

static Boolean HasDeviceProperty(const AudioObjectPropertyAddress* a) {
    switch (a->mSelector) {
        case kAudioObjectPropertyBaseClass:
        case kAudioObjectPropertyClass:
        case kAudioObjectPropertyOwner:
        case kAudioObjectPropertyName:
        case kAudioObjectPropertyManufacturer:
        case kAudioObjectPropertyOwnedObjects:
        case kAudioDevicePropertyDeviceUID:
        case kAudioDevicePropertyModelUID:
        case kAudioDevicePropertyTransportType:
        case kAudioDevicePropertyRelatedDevices:
        case kAudioDevicePropertyClockDomain:
        case kAudioDevicePropertyDeviceIsAlive:
        case kAudioDevicePropertyDeviceIsRunning:
        case kAudioDevicePropertyDeviceCanBeDefaultDevice:
        case kAudioDevicePropertyDeviceCanBeDefaultSystemDevice:
        case kAudioDevicePropertyLatency:
        case kAudioDevicePropertyStreams:
        case kAudioObjectPropertyControlList:
        case kAudioDevicePropertySafetyOffset:
        case kAudioDevicePropertyNominalSampleRate:
        case kAudioDevicePropertyAvailableNominalSampleRates:
        case kAudioDevicePropertyIsHidden:
        case kAudioDevicePropertyPreferredChannelsForStereo:
        case kAudioDevicePropertyPreferredChannelLayout:
        case kAudioDevicePropertyZeroTimeStampPeriod:
            return true;
        default:
            return false;
    }
}

static Boolean HasStreamProperty(const AudioObjectPropertyAddress* a) {
    switch (a->mSelector) {
        case kAudioObjectPropertyBaseClass:
        case kAudioObjectPropertyClass:
        case kAudioObjectPropertyOwner:
        case kAudioStreamPropertyIsActive:
        case kAudioStreamPropertyDirection:
        case kAudioStreamPropertyTerminalType:
        case kAudioStreamPropertyStartingChannel:
        case kAudioStreamPropertyLatency:
        case kAudioStreamPropertyVirtualFormat:
        case kAudioStreamPropertyPhysicalFormat:
        case kAudioStreamPropertyAvailableVirtualFormats:
        case kAudioStreamPropertyAvailablePhysicalFormats:
            return true;
        default:
            return false;
    }
}

static Boolean CM_HasProperty(AudioServerPlugInDriverRef inDriver, AudioObjectID inObjectID, pid_t inClientProcessID, const AudioObjectPropertyAddress* inAddress) {
    (void)inDriver; (void)inClientProcessID;
    switch (inObjectID) {
        case kObjectID_PlugIn: return HasPlugInProperty(inAddress);
        case kObjectID_Device: return HasDeviceProperty(inAddress);
        case kObjectID_Stream_Input:
        case kObjectID_Stream_Output: return HasStreamProperty(inAddress);
        default: return false;
    }
}

static OSStatus CM_IsPropertySettable(AudioServerPlugInDriverRef inDriver, AudioObjectID inObjectID, pid_t inClientProcessID, const AudioObjectPropertyAddress* inAddress, Boolean* outIsSettable) {
    (void)inDriver; (void)inObjectID; (void)inClientProcessID; (void)inAddress;
    *outIsSettable = false;
    return noErr;
}

static OSStatus CM_GetPropertyDataSize(AudioServerPlugInDriverRef inDriver, AudioObjectID inObjectID, pid_t inClientProcessID, const AudioObjectPropertyAddress* inAddress, UInt32 inQualifierDataSize, const void* inQualifierData, UInt32* outDataSize) {
    (void)inDriver; (void)inClientProcessID; (void)inQualifierDataSize; (void)inQualifierData;
    *outDataSize = 0;

    if (inObjectID == kObjectID_PlugIn) {
        switch (inAddress->mSelector) {
            case kAudioObjectPropertyBaseClass:
            case kAudioObjectPropertyClass:
            case kAudioObjectPropertyOwner:
                *outDataSize = sizeof(AudioClassID); return noErr;
            case kAudioObjectPropertyManufacturer:
            case kAudioPlugInPropertyResourceBundle:
                *outDataSize = sizeof(CFStringRef); return noErr;
            case kAudioObjectPropertyOwnedObjects:
            case kAudioPlugInPropertyDeviceList:
            case kAudioPlugInPropertyTranslateUIDToDevice:
                *outDataSize = sizeof(AudioObjectID); return noErr;
            default: return kAudioHardwareUnknownPropertyError;
        }
    }

    if (inObjectID == kObjectID_Device) {
        switch (inAddress->mSelector) {
            case kAudioObjectPropertyBaseClass:
            case kAudioObjectPropertyClass:
            case kAudioObjectPropertyOwner:
            case kAudioDevicePropertyTransportType:
            case kAudioDevicePropertyClockDomain:
            case kAudioDevicePropertyDeviceIsAlive:
            case kAudioDevicePropertyDeviceIsRunning:
            case kAudioDevicePropertyDeviceCanBeDefaultDevice:
            case kAudioDevicePropertyDeviceCanBeDefaultSystemDevice:
            case kAudioDevicePropertyLatency:
            case kAudioDevicePropertySafetyOffset:
            case kAudioDevicePropertyIsHidden:
            case kAudioDevicePropertyZeroTimeStampPeriod:
                *outDataSize = sizeof(UInt32); return noErr;
            case kAudioObjectPropertyName:
            case kAudioObjectPropertyManufacturer:
            case kAudioDevicePropertyDeviceUID:
            case kAudioDevicePropertyModelUID:
                *outDataSize = sizeof(CFStringRef); return noErr;
            case kAudioObjectPropertyOwnedObjects:
            case kAudioDevicePropertyStreams:
                if (inAddress->mScope == kAudioObjectPropertyScopeInput || inAddress->mScope == kAudioObjectPropertyScopeOutput) {
                    *outDataSize = sizeof(AudioObjectID);
                } else {
                    *outDataSize = 2 * sizeof(AudioObjectID);
                }
                return noErr;
            case kAudioObjectPropertyControlList:
                *outDataSize = 0; return noErr;
            case kAudioDevicePropertyRelatedDevices:
                *outDataSize = sizeof(AudioObjectID); return noErr;
            case kAudioDevicePropertyNominalSampleRate:
                *outDataSize = sizeof(Float64); return noErr;
            case kAudioDevicePropertyAvailableNominalSampleRates:
                *outDataSize = sizeof(AudioValueRange); return noErr;
            case kAudioDevicePropertyPreferredChannelsForStereo:
                *outDataSize = 2 * sizeof(UInt32); return noErr;
            case kAudioDevicePropertyPreferredChannelLayout:
                *outDataSize = offsetof(AudioChannelLayout, mChannelDescriptions) + kNumChannels * sizeof(AudioChannelDescription);
                return noErr;
            default: return kAudioHardwareUnknownPropertyError;
        }
    }

    if (inObjectID == kObjectID_Stream_Input || inObjectID == kObjectID_Stream_Output) {
        switch (inAddress->mSelector) {
            case kAudioObjectPropertyBaseClass:
            case kAudioObjectPropertyClass:
            case kAudioObjectPropertyOwner:
            case kAudioStreamPropertyIsActive:
            case kAudioStreamPropertyDirection:
            case kAudioStreamPropertyTerminalType:
            case kAudioStreamPropertyStartingChannel:
            case kAudioStreamPropertyLatency:
                *outDataSize = sizeof(UInt32); return noErr;
            case kAudioStreamPropertyVirtualFormat:
            case kAudioStreamPropertyPhysicalFormat:
                *outDataSize = sizeof(AudioStreamBasicDescription); return noErr;
            case kAudioStreamPropertyAvailableVirtualFormats:
            case kAudioStreamPropertyAvailablePhysicalFormats:
                *outDataSize = sizeof(AudioStreamRangedDescription); return noErr;
            default: return kAudioHardwareUnknownPropertyError;
        }
    }
    return kAudioHardwareBadObjectError;
}

static OSStatus CM_GetPropertyData(AudioServerPlugInDriverRef inDriver, AudioObjectID inObjectID, pid_t inClientProcessID, const AudioObjectPropertyAddress* inAddress, UInt32 inQualifierDataSize, const void* inQualifierData, UInt32 inDataSize, UInt32* outDataSize, void* outData) {
    (void)inDriver; (void)inClientProcessID;
    *outDataSize = 0;

    if (inObjectID == kObjectID_PlugIn) {
        switch (inAddress->mSelector) {
            case kAudioObjectPropertyBaseClass:
                if (inDataSize < sizeof(AudioClassID)) return kAudioHardwareBadPropertySizeError;
                *(AudioClassID*)outData = kAudioObjectClassID;
                *outDataSize = sizeof(AudioClassID); return noErr;
            case kAudioObjectPropertyClass:
                if (inDataSize < sizeof(AudioClassID)) return kAudioHardwareBadPropertySizeError;
                *(AudioClassID*)outData = kAudioPlugInClassID;
                *outDataSize = sizeof(AudioClassID); return noErr;
            case kAudioObjectPropertyOwner:
                if (inDataSize < sizeof(AudioObjectID)) return kAudioHardwareBadPropertySizeError;
                *(AudioObjectID*)outData = kAudioObjectUnknown;
                *outDataSize = sizeof(AudioObjectID); return noErr;
            case kAudioObjectPropertyManufacturer:
                if (inDataSize < sizeof(CFStringRef)) return kAudioHardwareBadPropertySizeError;
                *(CFStringRef*)outData = CFSTR(kDevice_Manufacturer);
                *outDataSize = sizeof(CFStringRef); return noErr;
            case kAudioPlugInPropertyResourceBundle:
                if (inDataSize < sizeof(CFStringRef)) return kAudioHardwareBadPropertySizeError;
                *(CFStringRef*)outData = CFSTR("");
                *outDataSize = sizeof(CFStringRef); return noErr;
            case kAudioObjectPropertyOwnedObjects:
            case kAudioPlugInPropertyDeviceList:
                if (inDataSize < sizeof(AudioObjectID)) return kAudioHardwareBadPropertySizeError;
                *(AudioObjectID*)outData = kObjectID_Device;
                *outDataSize = sizeof(AudioObjectID); return noErr;
            case kAudioPlugInPropertyTranslateUIDToDevice: {
                if (inDataSize < sizeof(AudioObjectID)) return kAudioHardwareBadPropertySizeError;
                AudioObjectID found = kAudioObjectUnknown;
                if (inQualifierDataSize == sizeof(CFStringRef) && inQualifierData != NULL) {
                    CFStringRef q = *(const CFStringRef*)inQualifierData;
                    if (q && CFEqual(q, CFSTR(kDevice_UID))) found = kObjectID_Device;
                }
                *(AudioObjectID*)outData = found;
                *outDataSize = sizeof(AudioObjectID); return noErr;
            }
            default: return kAudioHardwareUnknownPropertyError;
        }
    }

    if (inObjectID == kObjectID_Device) {
        switch (inAddress->mSelector) {
            case kAudioObjectPropertyBaseClass:
                *(AudioClassID*)outData = kAudioObjectClassID; *outDataSize = sizeof(AudioClassID); return noErr;
            case kAudioObjectPropertyClass:
                *(AudioClassID*)outData = kAudioDeviceClassID; *outDataSize = sizeof(AudioClassID); return noErr;
            case kAudioObjectPropertyOwner:
                *(AudioObjectID*)outData = kObjectID_PlugIn; *outDataSize = sizeof(AudioObjectID); return noErr;
            case kAudioObjectPropertyName:
                *(CFStringRef*)outData = CFSTR(kDevice_Name); *outDataSize = sizeof(CFStringRef); return noErr;
            case kAudioObjectPropertyManufacturer:
                *(CFStringRef*)outData = CFSTR(kDevice_Manufacturer); *outDataSize = sizeof(CFStringRef); return noErr;
            case kAudioDevicePropertyDeviceUID:
                *(CFStringRef*)outData = CFSTR(kDevice_UID); *outDataSize = sizeof(CFStringRef); return noErr;
            case kAudioDevicePropertyModelUID:
                *(CFStringRef*)outData = CFSTR(kDevice_ModelUID); *outDataSize = sizeof(CFStringRef); return noErr;
            case kAudioDevicePropertyTransportType:
                *(UInt32*)outData = kAudioDeviceTransportTypeVirtual; *outDataSize = sizeof(UInt32); return noErr;
            case kAudioDevicePropertyRelatedDevices:
                *(AudioObjectID*)outData = kObjectID_Device; *outDataSize = sizeof(AudioObjectID); return noErr;
            case kAudioDevicePropertyClockDomain:
                *(UInt32*)outData = 0; *outDataSize = sizeof(UInt32); return noErr;
            case kAudioDevicePropertyDeviceIsAlive:
                *(UInt32*)outData = 1; *outDataSize = sizeof(UInt32); return noErr;
            case kAudioDevicePropertyDeviceIsRunning:
                pthread_mutex_lock(&gStateMutex);
                *(UInt32*)outData = (gIOClientCount > 0) ? 1 : 0;
                pthread_mutex_unlock(&gStateMutex);
                *outDataSize = sizeof(UInt32); return noErr;
            case kAudioDevicePropertyDeviceCanBeDefaultDevice:
                // Allow as default INPUT microphone, NEVER hijack default OUTPUT speakers!
                *(UInt32*)outData = (inAddress->mScope == kAudioObjectPropertyScopeInput) ? 1 : 0;
                *outDataSize = sizeof(UInt32); return noErr;
            case kAudioDevicePropertyDeviceCanBeDefaultSystemDevice:
                *(UInt32*)outData = 0; *outDataSize = sizeof(UInt32); return noErr;
            case kAudioDevicePropertyLatency:
            case kAudioDevicePropertySafetyOffset:
                *(UInt32*)outData = 0; *outDataSize = sizeof(UInt32); return noErr;
            case kAudioDevicePropertyIsHidden:
                *(UInt32*)outData = 0; *outDataSize = sizeof(UInt32); return noErr;
            case kAudioDevicePropertyZeroTimeStampPeriod:
                *(UInt32*)outData = kZeroStampPeriod; *outDataSize = sizeof(UInt32); return noErr;
            case kAudioDevicePropertyNominalSampleRate:
                *(Float64*)outData = kSampleRate; *outDataSize = sizeof(Float64); return noErr;
            case kAudioDevicePropertyAvailableNominalSampleRates: {
                AudioValueRange r = { kSampleRate, kSampleRate };
                *(AudioValueRange*)outData = r; *outDataSize = sizeof(AudioValueRange); return noErr;
            }
            case kAudioObjectPropertyOwnedObjects:
            case kAudioDevicePropertyStreams: {
                AudioObjectID* list = (AudioObjectID*)outData;
                if (inAddress->mScope == kAudioObjectPropertyScopeInput) {
                    list[0] = kObjectID_Stream_Input; *outDataSize = sizeof(AudioObjectID);
                } else if (inAddress->mScope == kAudioObjectPropertyScopeOutput) {
                    list[0] = kObjectID_Stream_Output; *outDataSize = sizeof(AudioObjectID);
                } else {
                    if (inDataSize >= 2 * sizeof(AudioObjectID)) {
                        list[0] = kObjectID_Stream_Input;
                        list[1] = kObjectID_Stream_Output;
                        *outDataSize = 2 * sizeof(AudioObjectID);
                    }
                }
                return noErr;
            }
            case kAudioObjectPropertyControlList:
                *outDataSize = 0; return noErr;
            case kAudioDevicePropertyPreferredChannelsForStereo: {
                UInt32* ch = (UInt32*)outData;
                ch[0] = 1; ch[1] = 2; *outDataSize = 2 * sizeof(UInt32); return noErr;
            }
            case kAudioDevicePropertyPreferredChannelLayout: {
                UInt32 needed = offsetof(AudioChannelLayout, mChannelDescriptions) + kNumChannels * sizeof(AudioChannelDescription);
                if (inDataSize < needed) return kAudioHardwareBadPropertySizeError;
                AudioChannelLayout* lay = (AudioChannelLayout*)outData;
                memset(lay, 0, needed);
                lay->mChannelLayoutTag = kAudioChannelLayoutTag_UseChannelDescriptions;
                lay->mNumberChannelDescriptions = kNumChannels;
                lay->mChannelDescriptions[0].mChannelLabel = kAudioChannelLabel_Left;
                lay->mChannelDescriptions[1].mChannelLabel = kAudioChannelLabel_Right;
                *outDataSize = needed; return noErr;
            }
            default: return kAudioHardwareUnknownPropertyError;
        }
    }

    if (inObjectID == kObjectID_Stream_Input || inObjectID == kObjectID_Stream_Output) {
        Boolean isInput = (inObjectID == kObjectID_Stream_Input);
        switch (inAddress->mSelector) {
            case kAudioObjectPropertyBaseClass:
                *(AudioClassID*)outData = kAudioObjectClassID; *outDataSize = sizeof(AudioClassID); return noErr;
            case kAudioObjectPropertyClass:
                *(AudioClassID*)outData = kAudioStreamClassID; *outDataSize = sizeof(AudioClassID); return noErr;
            case kAudioObjectPropertyOwner:
                *(AudioObjectID*)outData = kObjectID_Device; *outDataSize = sizeof(AudioObjectID); return noErr;
            case kAudioStreamPropertyIsActive:
                *(UInt32*)outData = 1; *outDataSize = sizeof(UInt32); return noErr;
            case kAudioStreamPropertyDirection:
                *(UInt32*)outData = isInput ? 1 : 0; *outDataSize = sizeof(UInt32); return noErr;
            case kAudioStreamPropertyTerminalType:
                *(UInt32*)outData = isInput ? kAudioStreamTerminalTypeMicrophone : kAudioStreamTerminalTypeSpeaker;
                *outDataSize = sizeof(UInt32); return noErr;
            case kAudioStreamPropertyStartingChannel:
                *(UInt32*)outData = 1; *outDataSize = sizeof(UInt32); return noErr;
            case kAudioStreamPropertyLatency:
                *(UInt32*)outData = 0; *outDataSize = sizeof(UInt32); return noErr;
            case kAudioStreamPropertyVirtualFormat:
            case kAudioStreamPropertyPhysicalFormat:
                *(AudioStreamBasicDescription*)outData = MakeFormat();
                *outDataSize = sizeof(AudioStreamBasicDescription); return noErr;
            case kAudioStreamPropertyAvailableVirtualFormats:
            case kAudioStreamPropertyAvailablePhysicalFormats: {
                AudioStreamRangedDescription rd;
                rd.mFormat = MakeFormat();
                rd.mSampleRateRange.mMinimum = kSampleRate;
                rd.mSampleRateRange.mMaximum = kSampleRate;
                *(AudioStreamRangedDescription*)outData = rd;
                *outDataSize = sizeof(AudioStreamRangedDescription); return noErr;
            }
            default: return kAudioHardwareUnknownPropertyError;
        }
    }
    return kAudioHardwareBadObjectError;
}

static OSStatus CM_SetPropertyData(AudioServerPlugInDriverRef inDriver, AudioObjectID inObjectID, pid_t inClientProcessID, const AudioObjectPropertyAddress* inAddress, UInt32 inQualifierDataSize, const void* inQualifierData, UInt32 inDataSize, const void* inData) {
    (void)inDriver; (void)inObjectID; (void)inClientProcessID; (void)inAddress;
    (void)inQualifierDataSize; (void)inQualifierData; (void)inDataSize; (void)inData;
    return kAudioHardwareUnsupportedOperationError;
}

// MARK: - Real-Time Audio IO

static OSStatus CM_StartIO(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, UInt32 inClientID) {
    (void)inDriver; (void)inDeviceObjectID; (void)inClientID;
    pthread_mutex_lock(&gStateMutex);
    if (gIOClientCount == 0) {
        gNumberTimeStamps = 0;
        gAnchorHostTime = mach_absolute_time();
        gAnchorSampleTime = 0.0;
        os_unfair_lock_lock(&gRingLock);
        memset(gRingBuf, 0, sizeof(gRingBuf));
        gRingRead = 0;
        gRingWrite = 0;
        gRingCount = 0;
        gOutputActive = 0;
        os_unfair_lock_unlock(&gRingLock);
    }
    gIOClientCount++;
    pthread_mutex_unlock(&gStateMutex);
    return noErr;
}

static OSStatus CM_StopIO(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, UInt32 inClientID) {
    (void)inDriver; (void)inDeviceObjectID; (void)inClientID;
    pthread_mutex_lock(&gStateMutex);
    if (gIOClientCount > 0) gIOClientCount--;
    if (gIOClientCount == 0) {
        os_unfair_lock_lock(&gRingLock);
        memset(gRingBuf, 0, sizeof(gRingBuf));
        gRingRead = 0;
        gRingWrite = 0;
        gRingCount = 0;
        gOutputActive = 0;
        os_unfair_lock_unlock(&gRingLock);
    }
    pthread_mutex_unlock(&gStateMutex);
    return noErr;
}

static OSStatus CM_GetZeroTimeStamp(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, UInt32 inClientID, Float64* outSampleTime, UInt64* outHostTime, UInt64* outSeed) {
    (void)inDriver; (void)inDeviceObjectID; (void)inClientID;
    UInt64 now = mach_absolute_time();
    Float64 ticksPerCycle = gHostTicksPerFrame * (Float64)kZeroStampPeriod;
    UInt64 cycles = (now >= gAnchorHostTime) ? (UInt64)((Float64)(now - gAnchorHostTime) / ticksPerCycle) : 0;
    *outSampleTime = (Float64)(cycles * (UInt64)kZeroStampPeriod);
    *outHostTime = gAnchorHostTime + (UInt64)((Float64)cycles * ticksPerCycle);
    *outSeed = 1;
    return noErr;
}

static OSStatus CM_WillDoIOOperation(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, UInt32 inClientID, UInt32 inOperationID, Boolean* outWillDo, Boolean* outWillDoInPlace) {
    (void)inDriver; (void)inDeviceObjectID; (void)inClientID;
    Boolean willDo = (inOperationID == kAudioServerPlugInIOOperationReadInput ||
                      inOperationID == kAudioServerPlugInIOOperationWriteMix);
    *outWillDo = willDo;
    *outWillDoInPlace = true;
    return noErr;
}

static OSStatus CM_BeginIOOperation(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, UInt32 inClientID, UInt32 inOperationID, UInt32 inIOBufferFrameSize, const AudioServerPlugInIOCycleInfo* inIOCycleInfo) {
    (void)inDriver; (void)inDeviceObjectID; (void)inClientID;
    (void)inOperationID; (void)inIOBufferFrameSize; (void)inIOCycleInfo;
    return noErr;
}

static OSStatus CM_DoIOOperation(
    AudioServerPlugInDriverRef inDriver,
    AudioObjectID inDeviceObjectID,
    AudioObjectID inStreamObjectID,
    UInt32 inClientID,
    UInt32 inOperationID,
    UInt32 inIOBufferFrameSize,
    const AudioServerPlugInIOCycleInfo* inIOCycleInfo,
    void* ioMainBuffer,
    void* ioSecondaryBuffer
) {
    (void)inDriver; (void)inDeviceObjectID; (void)inStreamObjectID;
    (void)inClientID; (void)inIOCycleInfo; (void)ioSecondaryBuffer;
    if (!ioMainBuffer || inIOBufferFrameSize == 0) return noErr;

    float* samples = (float*)ioMainBuffer;

    if (inOperationID == kAudioServerPlugInIOOperationWriteMix) {
        os_unfair_lock_lock(&gRingLock);
        // Keep loopback latency minimal (~1-2 IO buffers) even if no reader was attached before
        UInt32 maxKeep = inIOBufferFrameSize * 2;
        if (maxKeep > kRingFrames - inIOBufferFrameSize) {
            maxKeep = kRingFrames - inIOBufferFrameSize;
        }
        if (gRingCount > maxKeep) {
            UInt32 drop = gRingCount - maxKeep;
            gRingRead = (gRingRead + drop) & kRingMask;
            gRingCount -= drop;
        }
        for (UInt32 f = 0; f < inIOBufferFrameSize; ++f) {
            UInt32 idx = gRingWrite * kNumChannels;
            gRingBuf[idx]     = samples[f * kNumChannels];
            gRingBuf[idx + 1] = samples[f * kNumChannels + 1];
            gRingWrite = (gRingWrite + 1) & kRingMask;
        }
        gRingCount += inIOBufferFrameSize;
        if (gRingCount > kRingFrames) {
            UInt32 overflow = gRingCount - kRingFrames;
            gRingRead = (gRingRead + overflow) & kRingMask;
            gRingCount = kRingFrames;
        }
        gOutputActive = 1;
        os_unfair_lock_unlock(&gRingLock);
    } else if (inOperationID == kAudioServerPlugInIOOperationReadInput) {
        os_unfair_lock_lock(&gRingLock);
        UInt32 take = (gRingCount < inIOBufferFrameSize) ? gRingCount : inIOBufferFrameSize;
        for (UInt32 f = 0; f < take; ++f) {
            UInt32 idx = gRingRead * kNumChannels;
            samples[f * kNumChannels]     = gRingBuf[idx];
            samples[f * kNumChannels + 1] = gRingBuf[idx + 1];
            gRingRead = (gRingRead + 1) & kRingMask;
        }
        gRingCount -= take;
        if (take < inIOBufferFrameSize) {
            memset(samples + take * kNumChannels, 0, (inIOBufferFrameSize - take) * kNumChannels * sizeof(float));
        }
        os_unfair_lock_unlock(&gRingLock);
    }
    return noErr;
}

static OSStatus CM_EndIOOperation(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, UInt32 inClientID, UInt32 inOperationID, UInt32 inIOBufferFrameSize, const AudioServerPlugInIOCycleInfo* inIOCycleInfo) {
    (void)inDriver; (void)inDeviceObjectID; (void)inClientID;
    (void)inOperationID; (void)inIOBufferFrameSize; (void)inIOCycleInfo;
    return noErr;
}
