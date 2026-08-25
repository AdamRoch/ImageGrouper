# Deploying the AutoHDR demo to AWS App Runner

The demo (`adamm13/autohdr-solution:v4` on Docker Hub, server entrypoint
`uvicorn server:app --host 0.0.0.0 --port 8080`) deploys to App Runner via ECR —
App Runner cannot pull from Docker Hub directly.

## Prerequisites

- aws CLI v2, configured for the target account (`aws configure`) with
  permissions for ECR, App Runner, and IAM role creation.
- Docker Desktop running locally (to re-tag and push the image).
- Nothing here runs until you say so — `deploy.sh` is the only executable step.

## 1. Deploy

```bash
./deploy/deploy.sh [REGION] [SERVICE_NAME] [IMAGE_TAG]
# e.g.  ./deploy/deploy.sh us-east-1 autohdr-demo v4
```

Idempotent: creates the ECR repo, the `AppRunnerECRAccessRole` IAM role, the
auto-scaling config (min 1 / max 1 — one instance, which also serializes
grouping jobs), and the App Runner service if missing; re-runs just push a new
image and trigger a deployment. Prints the default `*.awsapprunner.com` URL.

The service runs 2 vCPU / 4 GB (App Runner sizes are fixed presets). Grouping
~100 images takes a few minutes there; jobs are one-at-a-time by design (HTTP
429 when busy).

## 2. Custom subdomain (placeholder — you choose)

Pick a subdomain of adamroch.com, e.g. **`autohdr.adamroch.com`** (replace
everywhere below).

```bash
aws apprunner associate-custom-domain \
  --region us-east-1 \
  --service-arn <SERVICE_ARN from deploy output or console> \
  --domain-name autohdr.adamroch.com
```

The call returns **DNS validation CNAME records** (plus the target record
pointing at the App Runner URL). Add them at your DNS provider:

- **If adamroch.com is hosted on Route 53**: create the validation CNAMEs and
  the alias/ CNAME record in the hosted zone — App Runner validates
  automatically (usually minutes to ~an hour).
- **If DNS is external** (Cloudflare, Namecheap, …): add the same CNAMEs
  there. Mind proxy modes — validation records must resolve publicly, so
  keep them DNS-only (e.g. Cloudflare "grey cloud") until validated.

Certificate issuance + propagation can take from ~30 minutes to a few hours.
Status: `aws apprunner describe-custom-domains --service-arn ...`.

## Cost honesty (us-east-1 list prices, ballpark)

- App Runner bills **per vCPU-hour while the service processes requests**
  (~$0.064/vCPU-hr) plus a lower **provisioned** rate when idle (~$0.007/GB-hr
  memory). With 2 vCPU / 4 GB and light demo traffic expect roughly
  **$10–25/month** if left running; near-zero traffic still bills provisioned
  capacity (~$2–5/month).
- **Pause when not demoing**: `aws apprunner pause-service --service-arn ...`
  stops compute billing entirely (resume is `resume-service`, ~1 min cold
  start). This is the right default between demos.
- ECR storage: ~1 GB image ≈ **$0.10/month**. Data transfer out is negligible
  at demo volumes.
- Tearing down entirely (below) drops cost to ~$0.

## 3. Teardown

```bash
aws apprunner delete-service --region us-east-1 --service-arn <SERVICE_ARN>
aws ecr delete-repository --region us-east-1 --repository-name autohdr-solution --force
# optional: detach + delete AppRunnerECRAccessRole, and the <SERVICE_NAME>-asg
# auto-scaling configuration (both are harmless to leave)
```

Remember to also remove the custom-domain DNS records you added.
