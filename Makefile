# the compiler to use.
CC=g++
# options to pass to the compiler.
CFLAGS=-c -Wall -O3 -funroll-loops -funroll-all-loops
# options to pass to the linker
LDFLAGS=`pkg-config --cflags --libs opencv` -lm
# the soure files
SOURCES=mdb1.cpp
# the object files
OBJECTS=$(SOURCES:.c=.o)
# name of the executable
EXECUTABLE=mdb1

all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -rf *o $(EXECUTABLE) 

