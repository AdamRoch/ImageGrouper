#!/usr/bin/env bash
# Deploy the AutoHDR demo (adamm13/autohdr-solution:v4) to AWS App Runner via ECR.
# Idempotent: creates the ECR repo, IAM access role, and App Runner service only
# when missing; otherwise updates the service to the new image.
#
# Prerequisites: aws CLI v2 configured (aws configure) with permissions for ECR,
# App Runner, and IAM role creation. No resources are created until you run this.
#
# Usage:
#   ./deploy.sh [REGION] [SERVICE_NAME] [IMAGE_TAG]
# Example:
#   ./deploy.sh us-east-1 autohdr-demo v4

set -euo pipefail

REGION="${1:-us-east-1}"
SERVICE_NAME="${2:-autohdr-demo}"
IMAGE_TAG="${3:-v4}"

SRC_IMAGE="adamm13/autohdr-solution:${IMAGE_TAG}"
REPO_NAME="autohdr-solution"
ROLE_NAME="AppRunnerECRAccessRole"

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}"
IMAGE_URI="${ECR_URI}:${IMAGE_TAG}"
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"

echo "==> Region ${REGION} | service ${SERVICE_NAME} | image ${IMAGE_URI}"

# --- 1. ECR repository ---
if ! aws ecr describe-repositories --region "$REGION" --repository-names "$REPO_NAME" >/dev/null 2>&1; then
  echo "==> creating ECR repository ${REPO_NAME}"
  aws ecr create-repository --region "$REGION" --repository-name "$REPO_NAME" >/dev/null
fi

# --- 2. Tag + push the public Docker Hub image to ECR ---
echo "==> logging in to ECR"
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
echo "==> pushing ${IMAGE_URI}"
docker tag "$SRC_IMAGE" "$IMAGE_URI"
docker push "$IMAGE_URI"

# --- 3. IAM role App Runner assumes to pull from ECR ---
if ! aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  echo "==> creating IAM role ${ROLE_NAME}"
  aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "build.apprunner.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }' >/dev/null
  aws iam attach-role-policy --role-name "$ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess
fi

# --- 3b. Auto-scaling config: fixed at 1 instance (demo box; also serializes jobs) ---
ASG_NAME="${SERVICE_NAME}-asg"
ASG_ARN=$(aws apprunner list-auto-scaling-configurations --region "$REGION" \
  --query "AutoScalingConfigurationSummaryList[?AutoScalingConfigurationName=='${ASG_NAME}'].AutoScalingConfigurationArn | [0]" \
  --output text 2>/dev/null || true)
if [ -z "$ASG_ARN" ] || [ "$ASG_ARN" = "None" ]; then
  echo "==> creating auto-scaling configuration ${ASG_NAME} (min 1 / max 1)"
  ASG_ARN=$(aws apprunner create-auto-scaling-configuration --region "$REGION" \
    --auto-scaling-configuration-name "$ASG_NAME" \
    --min-size 1 --max-size 1 --max-concurrency 50 \
    --query 'AutoScalingConfiguration.AutoScalingConfigurationArn' --output text)
fi

# --- 4. Create or update the App Runner service ---
JSON_FILE="$(cd "$(dirname "$0")" && pwd)/apprunner-service.json"
render() {
  sed -e "s|__IMAGE_URI__|${IMAGE_URI}|g" \
      -e "s|__SERVICE_NAME__|${SERVICE_NAME}|g" \
      -e "s|__ROLE_ARN__|${ROLE_ARN}|g" \
      -e "s|__ASG_ARN__|${ASG_ARN}|g" "$JSON_FILE"
}

if SERVICE_ARN=$(aws apprunner list-services --region "$REGION" \
      --query "ServiceSummaryList[?ServiceName=='${SERVICE_NAME}'].ServiceArn | [0]" \
      --output text 2>/dev/null) && [ "$SERVICE_ARN" != "None" ] && [ -n "$SERVICE_ARN" ]; then
  echo "==> service exists (${SERVICE_ARN}); starting an update deployment"
  aws apprunner start-deployment --region "$REGION" --service-arn "$SERVICE_ARN" >/dev/null
else
  echo "==> creating App Runner service ${SERVICE_NAME}"
  render | aws apprunner create-service --region "$REGION" --cli-input-json file:///dev/stdin >/dev/null
fi

SERVICE_ARN=$(aws apprunner list-services --region "$REGION" \
  --query "ServiceSummaryList[?ServiceName=='${SERVICE_NAME}'].ServiceArn | [0]" --output text)
DOMAIN=$(aws apprunner describe-service --region "$REGION" --service-arn "$SERVICE_ARN" \
  --query 'Service.ServiceUrl' --output text)

echo "==> done. Service URL (may take a few minutes to go live): https://${DOMAIN}"
echo "    next: custom subdomain — see deploy/README.md"
