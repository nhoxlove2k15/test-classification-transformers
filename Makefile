setenv:
	source ics-env/bin/activate
api-lib:
	pip3 install fastapi uvicorn urlopen
model-lib:
	pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
service:
	uvicorn main:app --host 0.0.0.0 --port 8001
build:
	docker build . -t imgclassify:0.1
run:
	docker run -d --name imgclassifycontainer -p 8001:80 imgclassify:0.1