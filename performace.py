import ocular
from kubernetes import client

api_client = client.api_client.ApiClient(configuration='c29d119df3b14fb7a82207f29c8a2156c505a5948f3e4dcba6229c92b35c9006')

for pod_state in ocular.monitor(api_client, 
                                namespace='polyaxon', 
                                container_names=('plx-notebook-be90630d9d0740ada845276f0e3f70a4-749dc96cd-zr29h',), 
                                label_selector='app in (workers,dashboard),type=runner'):
    print(pod_state)