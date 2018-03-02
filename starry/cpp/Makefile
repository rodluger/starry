.PHONY : clean, default, coverage, profile
SOURCES = $(wildcard *.cpp)
HEADERS = $(wildcard *.h)

default: test

test: $(SOURCES) $(HEADERS)
	@g++ -std=c++11 -Wall -I/usr/local/include -c *.cpp
	@g++ -std=c++11 -L/usr/local/lib *.o -lgsl -lgslcblas -lm -o test
	@./test

profile: $(SOURCES) $(HEADERS)
	@g++ -std=c++11 -Wall -I/usr/local/include -c *.cpp
	@g++ -std=c++11 -L/usr/local/lib *.o -lgsl -lgslcblas -lm -o test
	@valgrind --tool=callgrind --callgrind-out-file=test.callgrind ./test
	@qcachegrind test.callgrind

coverage: $(SOURCES) $(HEADERS)
	@g++ --coverage -std=c++11 -Wall -I/usr/local/include -c *.cpp
	@g++ --coverage -std=c++11 -L/usr/local/lib *.o -lgsl -lgslcblas -lm -o test
	@./test
	@mkdir -p cov
	@mv *.gcda cov/
	@mv *.gcno cov/
	@cd cov
	@lcov -t "test" -o test.info -c -d .
	@genhtml -o cov/html test.info

clean:
	@rm -rf *.o
	@rm -rf cov
	@rm -rf test
	@rm -rf test.dSYM
	@rm -rf test.callgrind
	@rm -rf test.info
