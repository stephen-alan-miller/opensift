#LDFLAGS = -L/usr/local/Cellar/cairo/1.12.10/lib
#CPPFLAGS = -I/usr/local/Cellar/cairo/1.12.10/include

CC	= gcc
CFLAGS	+= -O3
BIN_DIR	= ./bin
SRC_DIR	= ./src
DOC_DIR	= ./docs
INC_DIR	= ./include
LIB_DIR	= ./lib
BIN	= siftfeat match dspfeat

all: $(BIN) libopensift.a #docs

docs:
	doxygen Doxyfile

libopensift.a:
	make -C $(SRC_DIR) $@

$(BIN):
	make -C $(SRC_DIR) $@

clean:
	make -C $(SRC_DIR) $@;	\
	make -C $(INC_DIR) $@;	\

distclean: clean
	rm -f $(LIB_DIR)/*
	rm -f $(BIN_DIR)/*

docsclean:
	rm -rf $(DOC_DIR)/html/

.PHONY: docs clean docsclean libopensift.a