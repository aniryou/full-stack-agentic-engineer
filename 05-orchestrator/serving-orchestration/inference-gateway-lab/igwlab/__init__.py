"""igwlab — an inference-gateway lab: route LLM requests across engine replicas like the llm-d EPP.

    igwlab.router       the router: prefix index, vLLM metric scraping, filters -> scorers -> picker, streaming proxy
    igwlab.fakebackend  an OpenAI-compatible backend with vLLM-shaped (emulated) timing, prefix cache and metrics
    igwlab.stack        LocalStack: N backends + router in-process on free ports (T0), or the router alone
                        in front of servers already running, e.g. real vLLM (T1)
    igwlab.bench        shared-prefix agentic workload; compare routing policies
    igwlab.autoscale    the Kubernetes HPA recommender, exactly; HPA manifests; targets from Little's law
    igwlab.k8s          manifest builders (InferencePool, InferenceObjective, HTTPRoute, ...) + CRD schema checks
    igwlab.promtext     Prometheus text parsing / rendering / histogram_quantile
"""
__version__ = "0.1.0"
