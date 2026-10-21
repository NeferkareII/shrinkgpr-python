# Handlers for the multivariate TPR (cases in compare/cases_mvtpr.py). The model,
# prediction and training handlers are shared with the MVGPR in handlers_mvgpr.R.

handlers$mv_ldt <- function(inp) {
  y <- D(inp$y)
  m <- mv_dummy("MVTPR", y)
  K <- D(inp$K)
  list(ll = m$ldt(K, D(inp$L_Om), torch_zeros(K$size(1), dtype = dtype), D(inp$nu)))
}

handlers$mv_ldg <- function(inp) {
  m <- mv_dummy("MVTPR", torch_randn(4, 2))
  list(ldg = m$ldg(D(inp$x), S(inp$alpha), S(inp$beta)))
}
