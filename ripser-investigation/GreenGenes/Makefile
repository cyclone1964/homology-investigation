build: createLowerDistanceMatrix

all: createLowerDistanceMatrix

createLowerDistanceMatrix: createLowerDistanceMatrix.c
	gcc -o $@ $< -Ofast 

clean:
	rm -f createLowerDistanceMatrix
