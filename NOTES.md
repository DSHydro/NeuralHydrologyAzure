# Comparison of GPUs

For now, focus on VMs with min specs & single GPU

| GPU | Architecture | Announcement / Release | Azure VM Series | Smallest VM SKU | vCPU | RAM (GB) | Cost per hour (USD) |
|---|---:|---|---|---|---|---|---|
| NVIDIA T4 | Turing | Sep 2018 (GA Q4 2018) | NCADS T4  | NC4as_T4_v3 | 4 | 28 | 0.53 |
| NVIDIA A100 | Ampere | May 2020 (mid‑2020 availability) | NCADS A100 | NC24ads_A100_v4 | 24 | 225 | 3.67 |
| NVIDIA L4/40S | Ada/Lovelace | Mar 2023 (announced / 2023 availability) |  ? | ? | ? | ? | ? |
| NVIDIA H100 | Hopper | Mar 2022 (announced / 2022 availability) | NCADS H100 | NC40ads_H100_v5 | 40 | 320 | 6.98 |
| NVIDIA B100/200 | Blackwell | Announced 2024 (availability post‑2024-06) | ND GB200 v6| ND-GB200-v6 | ? | ? | ? |

ND-series (Deep Learning Focus), NC-series (HPC/General ML Focus), NV-series (Visualization Focus)

# Azure regions

https://learn.microsoft.com/en-us/azure/reliability/regions-list

Washington State = westus2 (Washington State)
Microsoft Planetary Computer = westeurope (Netherlands)


"Standard NCADS_A100_v4 Family vCPUs are high in demand in westus2 for AI for Good Sponsorship. Consider alternative VM-series or regions. If you still want to continue, file a new support request and expect some delays."

## Mounting Azure resources to the runner

Or launching runners in a specific region?

https://docs.github.com/en/organizations/managing-organization-settings/configuring-private-networking-for-github-hosted-runners-in-your-organization


### OIDC Connect

These instructions seem most up-to-date:

https://github.com/Azure/login?tab=readme-ov-file#login-with-openid-connect-oidc-recommended

Compared to https://docs.github.com/en/actions/how-tos/secure-your-work/security-harden-deployments/oidc-in-azure


Apparently "login with a service principal secret" works for both GitHub-hosted and self-hosted runners... but "user or system-assigned managed identies" only work on self-hosted runners.

So follow these instructions:

https://github.com/Azure/login?tab=readme-ov-file#login-with-a-service-principal-secret

Test uploading a file to storage container
```bash
az storage blob upload \
    --account-name dshydro\
    --container-name accelerator2025 \
    --name README.md \
    --file README.md
```

```bash
az storage blob download \
    --account-name dshydro\
    --container-name accelerator2025 \
    --name README.md \
    --file /tmp/README.md
```

```bash
az storage blob list \
    --account-name dshydro \
    --container-name accelerator2025 \
    --output table \
    --auth-mode login
```

Test locally

```
az login --service-principal -u XXXX --tenant XXXX -p XXXX --allow-no-subscriptions
```

Check role permissions
```
az role assignment list --assignee-object-id 0e4b1a91-dc85-4f8f-822e-561eb0ce806e  --all --include-inherited -o table
```

