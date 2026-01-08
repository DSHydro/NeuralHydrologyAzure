Attempt to run NeuralHydrology in same region as our Azure account

https://docs.github.com/en/organizations/managing-organization-settings/configuring-private-networking-for-github-hosted-runners-in-your-organization

Org 'databaseID' seems to be the same as just 'id'
https://api.github.com/orgs/dshydro

1. Modify `./configure_vnet.sh`

```bash
./configure_vnet.sh
```

This outputs something like the following, you need `AAABBBBCCCDDD` for the next step on GitHub.
```
Create network settings resource github-actions-network-settings
GitHubId                                                          Name
----------------------------------------------------------------  -------------------------------
AAABBBBCCCDDD  github-actions-network-settings
```

Once runner groups are connected to the account VPC, I think they can communicate with other resources in that VPC, e.g., blob storage?
