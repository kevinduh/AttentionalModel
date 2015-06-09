CC=g++
CNN_DIR = /home/austinma/git/cnn
EIGEN = /opt/tools/eigen-dev/
CNN_BUILD_DIR=$(CNN_DIR)/build
INCS=-I$(CNN_DIR) -I$(CNN_BUILD_DIR) -I$(EIGEN)
LIBS=-L$(CNN_BUILD_DIR)/cnn/
FINAL=-lcnn -lboost_regex -lboost_serialization
CFLAGS=-std=c++11 -O0 -g -ffast-math -funroll-loops
BINDIR=bin
SRCDIR=src

.PHONY: clean
all: $(BINDIR)/lstmlm $(BINDIR)/train $(BINDIR)/predict $(BINDIR)/sandbox

$(BINDIR)/sandbox: src/sandbox.cc src/utils.h
	mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $(SRCDIR)/sandbox.cc -o $(BINDIR)/sandbox $(FINAL)

$(BINDIR)/train: $(BINDIR)/train.o $(BINDIR)/attentional.o $(BINDIR)/bitext.o
	mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $(BINDIR)/train.o $(BINDIR)/attentional.o $(BINDIR)/bitext.o -o $(BINDIR)/train $(FINAL)

$(BINDIR)/predict: $(BINDIR)/predict.o $(BINDIR)/attentional.o $(BINDIR)/bitext.o
	mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $(BINDIR)/predict.o $(BINDIR)/attentional.o $(BINDIR)/bitext.o -o $(BINDIR)/predict $(FINAL)

$(BINDIR)/train.o: $(SRCDIR)/train.cc $(SRCDIR)/attentional.h $(SRCDIR)/bitext.h
	mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $(INCS) -c $(SRCDIR)/train.cc -o $(BINDIR)/train.o

$(BINDIR)/predict.o: $(SRCDIR)/predict.cc $(SRCDIR)/attentional.h
	mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $(INCS) -c $(SRCDIR)/predict.cc -o $(BINDIR)/predict.o

$(BINDIR)/attentional.o: $(SRCDIR)/attentional.cc $(SRCDIR)/utils.h $(SRCDIR)/attentional.h $(SRCDIR)/bitext.h
	mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $(INCS) -c $(SRCDIR)/attentional.cc -o $(BINDIR)/attentional.o

$(BINDIR)/bitext.o: $(SRCDIR)/bitext.cc $(SRCDIR)/bitext.h
	mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $(INCS) -c $(SRCDIR)/bitext.cc -o $(BINDIR)/bitext.o

$(BINDIR)/train.o:


$(BINDIR)/lstmlm: src/lstmlm.cc src/utils.h
	mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $(SRCDIR)/lstmlm.cc -o $(BINDIR)/lstmlm $(FINAL)

clean:
	rm -rf $(BINDIR)/*
