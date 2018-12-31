HEADERS = $(wildcard ../../include/*.h)
SOURCES = $(wildcard *.cpp)
EXECUTABLES = $(SOURCES:.cpp=.test)

all: $(EXECUTABLES)

%.test: %.cpp $(HEADERS)
	@echo "Compiling $<..."
	@g++ -std=c++11 -g -Wall -O0 -I../../include -I../../lib/eigen_3.3.5 -I../../lib/boost_1_66_0 -I. $< -o $(<F:.cpp=.test)

clean:
	@rm -rf *.test
	@rm -rf *.dSYM