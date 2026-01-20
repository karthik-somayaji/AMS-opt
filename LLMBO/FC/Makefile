CC=g++

TARGET=fc
SOURCE=fc.cpp
CFLAGS=-fopenmp -lpthread -std=c++11

all: ${TARGET}

${TARGET}: ${SOURCE}
	${CC} $^ -o $@ ${CFLAGS}

.PHONY: clean
clean: 
	rm -f ${TARGET}
