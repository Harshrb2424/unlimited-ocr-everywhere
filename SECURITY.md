# Security policy

## Supported versions

This is a community packaging project. Security fixes are applied to the `main`
branch. There are no long-term support branches.

## Reporting a vulnerability

Please do **not** open a public issue for a security problem.

Use GitHub's private vulnerability reporting:

1. Open the repository's **Security** tab.
2. Click **Report a vulnerability**.
3. Include a description, reproduction steps, and affected files.

If private reporting is unavailable, contact the maintainer listed on the
repository profile.

## Scope

Relevant issues include:

- Arbitrary code execution through the notebooks, CLI, or post-processor
- Unsafe deserialization or path traversal in input handling
- Supply-chain issues in dependencies or the customized SGLang wheel
- Accidental upload of private documents

## Important context

- `AutoModel.from_pretrained(..., trust_remote_code=True)` executes Python code
  downloaded from the model repository. The notebooks do this by design. If you
  need a stricter posture, review the modeling files and pin a revision.
- The notebooks send document images to the model locally. On Google Colab,
  processing happens in Google's cloud. Do not upload confidential documents to
  a third-party runtime unless your policy allows it.
- The model downloads from Hugging Face. A compromised model repository is a
  supply-chain risk; pin a revision in high-security environments.
- The SGLang notebook downloads a wheel from the upstream GitHub repository.
  Verify the wheel hash before deploying it in production.

## Disclosure

We will acknowledge reports within a reasonable time and coordinate a fix and
disclosure. Please give us a chance to address the issue before publishing it.
