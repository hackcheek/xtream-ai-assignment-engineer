ch1:
	jupyter lab .

ch2: ch2_download_data ch2_run_pipeline

ch2_run_pipeline: ch2_pf_minio_pod
	python -m challenge2.src.pipelines.challenge2

ch2_download_data:
	bash challenge2/download_new_dataset.sh

ch2_pf_minio_pod:
	lsof -i :9000 | grep kubectl | head -n 1 | awk '{print $$2}' | xargs kill
	$(eval minio_pod_name = $(shell kubectl get pod -n kubeflow | grep -i minio | awk '{print $$1}'))
	kubectl port-forward -n kubeflow $(minio_pod_name) 9000:9000 &>/dev/null &

ch2_dashboard:
	bash -c "sleep 2; open http://localhost:8080" &
	kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80

start_kubeflow_cluster:
	kind create cluster
	k3d cluster create mycluster

	export PIPELINE_VERSION=2.1.0
	kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
	kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
	kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-emissary?ref=$PIPELINE_VERSION"

restart_kubeflow:
	kubectl rollout restart deployment/ml-pipeline-ui -n kubeflow
