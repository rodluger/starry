.PHONY : clean, default, memcheck
HEADERS = $(wildcard *.h)

default: test

test: test.cpp $(HEADERS)
	@g++ -std=c++11 -Wall -g -I/usr/local/include -I/Users/rodrigo/src test.cpp -o test
	@./test

memcheck: test.cpp $(HEADERS)
	@g++ -std=c++11 -Wall -g -I/usr/local/include -I/Users/rodrigo/src test.cpp -o test
	@valgrind --tool=memcheck ./test

clean:
	@rm -rf *.o
	@rm -rf test
	@rm -rf test.dSYM
	@rm -rf test.info
