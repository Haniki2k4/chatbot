#!/bin/bash
# Setup script for chatbot deployment preparation
# Run: bash setup.sh

set -e

echo "🚀 Chatbot Production Setup Script"
echo "═════════════════════════════════════"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Create directories
echo -e "${BLUE}Step 1: Creating directories...${NC}"
mkdir -p data/documents
mkdir -p data/cache
mkdir -p models
mkdir -p web/templates
mkdir -p web/static
mkdir -p tests
mkdir -p src
mkdir -p config
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Step 2: Environment setup
echo -e "${BLUE}Step 2: Setting up environment...${NC}"
if [ ! -f .env ]; then
    cp .env.example .env
    echo -e "${GREEN}✓ .env file created${NC}"
else
    echo -e "${YELLOW}⚠ .env already exists${NC}"
fi
echo ""

# Step 3: Create Python virtual environment
echo -e "${BLUE}Step 3: Creating Python virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}⚠ venv already exists${NC}"
fi
echo ""

# Step 4: Activate venv and install dependencies
echo -e "${BLUE}Step 4: Installing dependencies...${NC}"
if [ -d "venv" ]; then
    source venv/bin/activate 2>/dev/null || . venv/Scripts/activate 2>/dev/null
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠ Could not activate venv${NC}"
fi
echo ""

# Step 5: Check for documents
echo -e "${BLUE}Step 5: Checking for documents...${NC}"
DOC_COUNT=$(find data/documents -name "*.txt" 2>/dev/null | wc -l)
if [ $DOC_COUNT -gt 0 ]; then
    echo -e "${GREEN}✓ Found $DOC_COUNT documents${NC}"
else
    echo -e "${YELLOW}⚠ No documents found in data/documents/${NC}"
    echo "   Place your .txt files in data/documents/ before running"
fi
echo ""

# Step 6: Check for models
echo -e "${BLUE}Step 6: Checking for LLM models...${NC}"
MODEL_COUNT=$(find models -name "*.gguf" 2>/dev/null | wc -l)
if [ $MODEL_COUNT -gt 0 ]; then
    echo -e "${GREEN}✓ Found $MODEL_COUNT GGUF models${NC}"
else
    echo -e "${YELLOW}⚠ No GGUF models found in models/${NC}"
    echo "   Download models to enable LLM features, or set USE_LLM=False in .env"
fi
echo ""

# Step 7: Test imports
echo -e "${BLUE}Step 7: Testing Python imports...${NC}"
python -c "
try:
    from config import get_config
    print('✓ Config import OK')
    from src.chatbot_engine import DucGiangChatbot
    print('✓ Chatbot import OK')
    from web.app import app
    print('✓ Flask app import OK')
except Exception as e:
    print(f'✗ Import error: {e}')
    exit(1)
" && echo -e "${GREEN}✓ All imports successful${NC}" || echo -e "${YELLOW}⚠ Import check failed${NC}"
echo ""

# Step 8: Summary
echo "═════════════════════════════════════"
echo -e "${GREEN}✓ Setup complete!${NC}"
echo ""
echo "📚 Next steps:"
echo "  1. Read README.md for overview"
echo "  2. Place documents in data/documents/"
echo "  3. Download LLM model to models/ (optional)"
echo "  4. Edit .env with your settings"
echo "  5. Run: python app.py"
echo ""
echo "🚀 Ready for deployment!"
echo "   See DEPLOY.md for HF Spaces deployment guide"
echo ""
