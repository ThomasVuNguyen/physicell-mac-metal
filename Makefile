# ─────────────────────────────────────────────────────────────────────
# PhysiCell Metal — Makefile for Apple Silicon
# Builds Objective-C++ + C++ + Metal shaders into a single binary
# ─────────────────────────────────────────────────────────────────────

# Compiler settings
CXX       = clang++
CXXFLAGS  = -std=c++17 -O3 -Wall -Wextra -Wno-unused-parameter
OBJCFLAGS = -fobjc-arc
METALC    = xcrun -sdk macosx metal
METALLIB  = xcrun -sdk macosx metallib

# Frameworks
FRAMEWORKS = -framework Metal -framework Foundation -framework MetalKit

# Directories
SRC_DIR    = src
SHADER_DIR = shaders
LIB_DIR    = lib
BUILD_DIR  = build
OBJ_DIR    = $(BUILD_DIR)/obj

# Include paths
INCLUDES = -I$(SHADER_DIR) -I$(LIB_DIR)/pugixml -I$(SRC_DIR)

# Target
TARGET = $(BUILD_DIR)/physicell-metal

# ─── Source files ───
# Objective-C++ files (need Metal framework)
MM_SRCS = $(SRC_DIR)/main.mm \
          $(SRC_DIR)/metal_context.mm \
          $(SRC_DIR)/microenvironment.mm \
          $(SRC_DIR)/cell_mechanics.mm

# Pure C++ files
CPP_SRCS = $(SRC_DIR)/cell_data.cpp \
           $(SRC_DIR)/config_parser.cpp \
           $(SRC_DIR)/cell_phenotype.cpp \
           $(SRC_DIR)/cell_interactions.cpp \
           $(SRC_DIR)/geometry.cpp \
           $(SRC_DIR)/output_writer.cpp \
           $(SRC_DIR)/motility.cpp \
           $(SRC_DIR)/cell_definitions.cpp \
           $(SRC_DIR)/signals.cpp \
           $(SRC_DIR)/behaviors.cpp \
           $(SRC_DIR)/rules.cpp \
           $(SRC_DIR)/svg_writer.cpp \
           $(SRC_DIR)/matlab_writer.cpp \
           $(LIB_DIR)/pugixml/pugixml.cpp

# Metal shader files
METAL_SRCS = $(SHADER_DIR)/diffusion_2d.metal \
             $(SHADER_DIR)/mechanics.metal \
             $(SHADER_DIR)/integrate.metal

# ─── Object files ───
MM_OBJS  = $(patsubst $(SRC_DIR)/%.mm,$(OBJ_DIR)/%.o,$(filter $(SRC_DIR)/%,$(MM_SRCS)))
CPP_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(filter $(SRC_DIR)/%,$(CPP_SRCS)))
LIB_OBJS = $(OBJ_DIR)/pugixml.o

METAL_AIR = $(patsubst $(SHADER_DIR)/%.metal,$(BUILD_DIR)/%.air,$(METAL_SRCS))
METALLIB_OUT = $(BUILD_DIR)/shaders.metallib

ALL_OBJS = $(MM_OBJS) $(CPP_OBJS) $(LIB_OBJS)

# ─── Targets ───

.PHONY: all clean run

all: $(TARGET) try-metallib

# Try to compile Metal shaders (optional — runtime fallback exists)
.PHONY: try-metallib
try-metallib:
	@if xcrun --find metal >/dev/null 2>&1; then \
		$(MAKE) $(METALLIB_OUT); \
	else \
		echo "⚠️  Metal compiler not found (need full Xcode), using runtime shader compilation"; \
	fi

# Link everything
$(TARGET): $(ALL_OBJS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) -o $@ $^
	@echo "✅ Built $(TARGET)"

# Compile Metal shaders → .air → .metallib
$(BUILD_DIR)/%.air: $(SHADER_DIR)/%.metal $(SHADER_DIR)/types.h | $(BUILD_DIR)
	$(METALC) -c -I$(SHADER_DIR) -o $@ $<

$(METALLIB_OUT): $(METAL_AIR)
	$(METALLIB) -o $@ $^
	@echo "✅ Built Metal shader library"

# Compile Objective-C++ files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.mm | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(OBJCFLAGS) $(INCLUDES) -c -o $@ $<

# Compile C++ files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

# Compile pugixml
$(OBJ_DIR)/pugixml.o: $(LIB_DIR)/pugixml/pugixml.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -Wno-everything -c -o $@ $<

# Create directories
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Run
run: all
	@echo "──────────────────────────────────────"
	@echo "  PhysiCell Metal — Apple Silicon"
	@echo "──────────────────────────────────────"
	./$(TARGET) config/PhysiCell_settings.xml

# Clean
clean:
	rm -rf $(BUILD_DIR)

# Print info
info:
	@echo "Sources (C++): $(CPP_SRCS)"
	@echo "Sources (ObjC++): $(MM_SRCS)"
	@echo "Sources (Metal): $(METAL_SRCS)"
	@echo "Target: $(TARGET)"
