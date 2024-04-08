ch1:
	jupyter lab .

ch2: ch2_download_data ch2_run_pipeline

ch2_run_pipeline:
	python -m challenge2.src.pipelines.challenge2

ch2_download_data:
	bash challenge2/download_new_dataset.sh

ch2_dashboard:
	bash -c "sleep 2; open http://localhost:8080" &
	kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
