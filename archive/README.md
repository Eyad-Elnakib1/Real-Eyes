# RealEyes — Historical & Prototype Archive

This directory contains legacy code, early prototypes, and prior iterations of subsystems preserved for historical reference, benchmark comparison, and academic auditing.

## Contents

- **`factcheck_ai/`**: An earlier standalone prototype of the AI text fact-checking service built during the initial research phase. The active, production-grade fact-checking pipeline is now integrated into `backend/fact_checker/` and served via the main API gateway in `backend/server.py`.

> [!NOTE]
> Code in this directory is excluded from active CI/CD build pipelines and production deployments.
