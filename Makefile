VENV=venv
BIN=$(VENV)/bin

# make it work on windows too
ifeq ($(OS), Windows_NT)
    BIN=$(VENV)/Scripts
endif


$(BIN)/activate: requirements.txt
	python3 -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install --upgrade -r requirements.txt


init: $(BIN)/activate
	mkdir -p ./data/processed3/test
	mkdir -p ./data/processed3/trainAE
	mkdir -p ./data/processed3/trainDBN
	mkdir -p ./data/processed3/valAE
	mkdir -p ./data/processed3/valDBN
	mkdir -p ./data/raw


dataset: init
	$(BIN)/python ./preprocessing/cicids2017.py

zeroday: init
	$(BIN)/python ./preprocessing/cicids2017_2.py

onlyAttacks: init
	$(BIN)/python ./preprocessing/cicids2017onlyattacks.py

run: init
	$(BIN)/python main.py --config ./configs/deepBeliefNetwork.json
	$(BIN)/python main.py --config ./configs/multilayerPerceptron.json


clean:
	rm -rf __pycache__
	rm -rf $(VENV)
