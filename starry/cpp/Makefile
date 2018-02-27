.PHONY : clean, default
SOURCES = $(wildcard *.cpp)
HEADERS = $(wildcard *.h)

default: test

test: $(SOURCES) $(HEADERS)
	@g++ -Wall -I/usr/local/include -c *.cpp
	@g++ -L/usr/local/lib *.o -lgsl -lgslcblas -lm -o test
	@./test

clean:
	@rm -rf *.o
	@rm -rf test
