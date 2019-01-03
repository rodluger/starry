.PHONY: test clean memcheck
HEADERS = $(wildcard *.h) $(wildcard ../../include/*.h) $(wildcard ../../include/*/*.h) $(wildcard ../../include/*/*/*.h)

test: test.cpp $(HEADERS)
	@echo "Compiling tests..."
	@g++ -std=c++11 -g -Wall -O0 -I../../include -I../../lib/eigen_3.3.5 -I../../lib/boost_1_66_0 -I. test.cpp -o test

clean:
	@rm -rf test
	@rm -rf *.dSYM

memcheck:
	valgrind --leak-check=yes test