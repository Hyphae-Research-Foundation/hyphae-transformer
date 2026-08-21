# Attribution And Independence

Celiums ReZero is an independent clean implementation inspired by:

> Thomas Bachlechner, Bodhisattwa Prasad Majumder, Huanru Henry Mao, Garrison W.
> Cottrell, and Julian McAuley. "ReZero is All You Need: Fast Convergence at Large
> Depth." UAI 2021.

The canonical residual equation is:

```text
x_next = x + alpha * F(x), alpha(0) = 0
```

Celiums ReZero does not copy the original PyTorch 1.4 implementation. It reimplements
the mechanism against current PyTorch primitives and tests the original strategy next
to explicitly named extensions.

The Lab is methodologically informed by public work on Practical NLP, CLIN,
DiscoveryBench, CodeScientist, Skill Set Optimization, and AutoDiscovery. Those
projects are research references, not dependencies or endorsements.

Names belonging to other projects and institutions are used only for accurate
attribution. Celiums ReZero is not an official continuation of ReZero, Practical NLP,
Ai2, or Asta.
