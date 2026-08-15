We Propose TKV which resolves running Brand's update only through a bounded warmup window, then freezing the resulting basis and projecting subsequent tokens at O(Rd) per token indefinitely. 

I referenced Halko Cold Starts with a phi-bridge as a possaible technique. We used T_0 tokens then TKV initializes the SVD factorization via the randomized range finder.

