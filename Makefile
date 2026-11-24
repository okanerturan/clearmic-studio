.PHONY: native app run clean

all: app

native:
	clang++ -std=c++20 -O3 -mcpu=native -ffast-math -shared -fPIC \
		-framework Accelerate engine/native_dsp.cpp -o engine/libnative_dsp.dylib
	clang++ -ObjC++ -std=c++20 -O3 -mcpu=native -fobjc-arc -shared -fPIC \
		-framework Foundation -framework CoreML engine/coreml_ane.mm -o engine/libcoreml_ane.dylib

app: native
	chmod +x macos_app/build_app.sh
	./macos_app/build_app.sh

run: app
	open "ClearMic Studio.app"

clean:
	rm -rf "ClearMic Studio.app" engine/libnative_dsp.dylib engine/libcoreml_ane.dylib
