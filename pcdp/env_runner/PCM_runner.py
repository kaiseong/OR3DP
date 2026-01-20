from pcdp.policy.base_pointcloud_policy import BasePointCloudPolicy
from pcdp.env_runner.base_pointcloud_runner import BasePointCloudRunner

class PCMPointCloudRunner(BasePointCloudRunner):
    def __init__(self, output_dir):
        super().__init__(output_dir)
    
    def run(self, policy: BasePointCloudPolicy):
        return dict()