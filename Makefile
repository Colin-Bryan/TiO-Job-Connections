.PHONY: run run-container gcloud-deploy

run:
	@streamlit run app.py --server.port=8080 --server.address=0.0.0.0

run-container:
	@docker build . -t app.py
	@docker run -p 8080:8080 app.py

gcloud-deploy:
	@gcloud config set project tio-job-connections
	@gcloud app deploy app.yaml --stop-previous-version

