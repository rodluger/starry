.PHONY : clean, default, memcheck
HEADERS = $(wildcard *.h)

default: test

test_autodiff: test_autodiff.cpp ellip.h
	@g++ -std=c++11 -Wall -g -I../lib/eigen_3.3.3 test_autodiff.cpp -o test_autodiff
	@./test_autodiff

test: test.cpp $(HEADERS)
	@g++ -std=c++11 -Wall -g -I../lib/eigen_3.3.3 -I. test.cpp -o test
	@./test

memcheck: test.cpp $(HEADERS)
	@g++ -std=c++11 -Wall -g -I../lib/eigen_3.3.3 -I. test.cpp -o test
	@valgrind --tool=memcheck ./test

clean:
	@rm -rf *.o
	@rm -rf test
	@rm -rf test.dSYM
	@rm -rf test.info
