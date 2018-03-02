.PHONY : clean, default
SOURCES = $(wildcard *.cpp)
HEADERS = $(wildcard *.h)

default: test

test: $(SOURCES) $(HEADERS)
	@g++ -std=c++11 -Wall -I/usr/local/include -I/anaconda/include/python3.4m -c *.cpp
	@g++ -std=c++11 -L/usr/local/lib *.o -lgsl -lgslcblas -lm -o test
	@./test

clean:
	@rm -rf *.o
	@rm -rf test
