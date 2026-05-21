# Kubernetes Deployment Notes

## DonkeyCar training Job

1. Build the image from the repository root:

   ```bash
   docker build -f deploy/docker/Dockerfile.donkeycar -t mytorch-donkeycar:latest .
   ```

2. Load or push the image to your cluster, for example:

   ```bash
   minikube image load mytorch-donkeycar:latest
   ```

3. Prepare two PVCs:

   - `donkeycar-data-pvc`: mounted at `/data/donkeycar`, read-only. It should contain the DonkeyCar images plus `splits/temporal_block_gap20/train.txt` and `val.txt`.
   - `donkeycar-results-pvc`: mounted at `/outputs`, used for result files.

4. Submit the Job:

   ```bash
   kubectl apply -f deploy/k8s/donkeycar-training-job.yaml
   kubectl logs -f job/mytorch-donkeycar-train
   ```

## Legacy MNIST demo

`mnist-training-job.yaml` is kept as a small deployment smoke test. The defense
project should use `donkeycar-training-job.yaml`.
