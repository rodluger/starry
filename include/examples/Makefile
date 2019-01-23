.PHONY: example clean
HEADERS = $(wildcard *.h) $(wildcard ../include/starry2/*.h) $(wildcard ../include/starry2/*/*.h) $(wildcard ../include/starry2/*/*/*.h)

example: example.cpp $(HEADERS)
	@echo "Compiling tests..."
	@g++ -std=c++11 -g -Wall -O0 -I../starry2 -I../../lib/eigen_3.3.5 -I../../lib/boost_1_66_0 -I. example.cpp -o example

clean:
	@rm -rf example
	@rm -rf *.dSYM