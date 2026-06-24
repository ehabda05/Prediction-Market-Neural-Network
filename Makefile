CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -O2
SRCS = main.cpp NeuralNetwork.cpp
PYSRC = kalshi.py 
OBJS = $(SRCS:.cpp=.o)
EXEC = neural_network

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^
	python $(PYSRC)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(EXEC) *.csv