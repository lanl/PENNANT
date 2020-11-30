CXX=hipcc
CXXOPTS= -Wno-c++17-extensions


example: example.cpp pajama.cpp pajama.h
	${CXX} ${CXXOPTS} example.cpp pajama.cpp -o $@

clean:
	rm -f example *.o *~

