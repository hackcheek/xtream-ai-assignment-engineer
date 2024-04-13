### How to deploy it on cloud

- To deploy this API we just need to dokerize challenge3/ code and deploy this container into a EC2 module in AWS and expose this service with API GATEWAY module. \
- Another way is create a new pod in our kubernetes cluster and deploy this in GCP GKE (Google Kubernetes Engine) and expose with API GATEWAY. \
- The more scalable way is create a terraform file so this will deploy to every cloud platform
