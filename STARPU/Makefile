
CC = icc
CFLAGS += -std=gnu99  -g -march=native -O2 -Wall
LDLIBS  =  -lopenblas -llapack -lm
OUT     = str_QR_test
OBJECTS = task-factorize-seq.o task-factorize-par.o update_panel_seq.o update_panel_par.o task-update-seq.o task-update-par.o random_matrix.o print_matrix.o test_reduction.o spin-barrier.o utils.o

STARPU_LIBS = -lstarpu-1.2

default: main

main: main.o driver.o
	$(CC) $(CFLAGS) $^ $(OBJECTS) $(LDLIBS) $(STARPU_LIBS) -o $(OUT)

main.o: main.c driver.o
	$(CC) $(CFLAGS) -c main.c

driver.o: driver.c $(OBJECTS) 
	$(CC) $(CFLAGS) -c driver.c

%.o: ../src/%.c
	$(CC) -c -o $@ $< $(CFLAGS)

.PHONY: clean

clean:
	rm -f $(OUT) *o

