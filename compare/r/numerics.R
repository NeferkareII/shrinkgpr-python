# Evaluates the R implementation on the cases written by compare/numerics.py.
# Usage: Rscript numerics.R <cases.json> <results.json>

args <- commandArgs(trailingOnly = TRUE)
source(file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)))), "common.R"))

input <- read_input(args[1])
dtype <- set_precision(input$dtype)
D <- function(x) dec(x, dtype)
# R torch creates float32 from R doubles whatever the default dtype, so scalars
# need an explicit dtype
S <- function(x) torch_tensor(x, dtype = dtype)
jit <- getFromNamespace(".shrinkGPR_internal", "shrinkGPR")$jit_funcs
cpu <- torch_device("cpu")

first <- function(x) if (is.list(x)) x[[1]] else x

# A GPR_class whose gen_batch replays a list of base samples in order
set_replay <- function(zs) {
  replay <- new.env()
  replay$i <- 0
  assign(".replay", list(zs = zs, env = replay), envir = globalenv())
  replay
}
GPR_replay <- nn_module(
  inherit = shrinkGPR:::GPR_class,
  gen_batch = function(n_latent) {
    r <- get(".replay", envir = globalenv())
    r$env$i <- r$env$i + 1
    r$zs[[r$env$i]]
  }
)

new_gpr <- function(inp, module_class = shrinkGPR:::GPR_class) {
  m <- module_class(D(inp$y), D(inp$x), D(inp$x_mean),
                    a = inp$a, c = inp$c, a_mean = inp$a_mean, c_mean = inp$c_mean,
                    sigma2_rate = inp$sigma2_rate, n_layers = inp$n_layers,
                    flow_func = flows[[inp$flow]], flow_args = inp$flow_args,
                    kernel_func = kernels[[inp$kernel]], device = cpu)
  # The prior buffers are created as float32; cast everything to the precision
  m$to(dtype = dtype)
  m
}

handlers <- list(
  hyperu = function(inp) {
    list(u = shrinkGPR:::robust_hyperu(S(inp$a), S(inp$b), D(inp$z)))
  },

  log_hyperu = function(inp) {
    z <- D(inp$z)$requires_grad_(TRUE)
    out <- first(shrinkGPR:::log_hyperu(S(inp$a), S(inp$b), z))
    out$sum()$backward()
    list(value = out, grad = z$grad)
  },

  lightweight_ols = function(inp) {
    list(p = shrinkGPR:::lightweight_ols(as.numeric(unlist(inp$y$data)),
                                         as.numeric(unlist(inp$x$data))))
  },

  robust_chol = function(inp) list(L = shrinkGPR:::robust_chol(D(inp$A))),

  res_protector = function(inp) {
    list(out = shrinkGPR:::res_protector_autograd(D(inp$x), tol = inp$tol))
  },

  sqdist = function(inp) list(D = shrinkGPR:::sqdist(D(inp$x), D(inp$thetas), D(inp$x_star))),

  kernel = function(inp) {
    list(K = kernels[[inp$kernel]](D(inp$thetas), D(inp$tau), D(inp$x), D(inp$x_star)))
  },

  flow = function(inp) {
    flow <- do.call(flows[[inp$flow]], c(list(d = inp$d), inp$args))
    load_state(flow, inp$state, dtype)
    out <- flow(D(inp$z))
    (out$zk$sum() + out$log_diag_j$sum())$backward()
    c(list(zk = out$zk, log_diag_j = out$log_diag_j), grads_of(flow))
  },

  priors = function(inp) {
    m <- shrinkGPR:::GPR_class(torch_randn(3), torch_randn(3, 1), NULL, n_layers = 1,
                               flow_func = sylvester, flow_args = list(), device = cpu)
    a <- S(inp$a); c <- S(inp$c)
    list(ltg = m$ltg(D(inp$theta), a, c, D(inp$tau)),
         ngg = m$ngg(D(inp$beta), a, c, D(inp$tau)),
         ldf = m$ldf(D(inp$tau), 2 * c, 2 * a),
         lexp = m$lexp(D(inp$tau), S(inp$rate)))
  },

  ldnorm = function(inp) {
    list(ll = jit$ldnorm(K = D(inp$K), L = NULL, sigma2 = D(inp$sigma2), y = D(inp$y),
                         x_mean = D(inp$x_mean), beta = D(inp$beta)))
  },

  gpr = function(inp) {
    m <- new_gpr(inp)
    load_state(m, inp$state, dtype)
    out <- m(D(inp$z))
    elbo <- m$elbo(out$zk, out$log_det_J)
    (-elbo)$backward()
    c(list(zk = out$zk, log_det_J = out$log_det_J, elbo = elbo), grads_of(m))
  },

  pred = function(inp) {
    replay <- set_replay(rep(list(D(inp$z)), 3))
    m <- new_gpr(inp, GPR_replay)
    load_state(m, inp$state, dtype)
    x_new <- D(inp$x_new)
    x_mean_new <- D(inp$x_mean_new)
    moments <- m$calc_pred_moments(x_new, inp$nsamp, x_mean_new)
    torch_manual_seed(inp$seed)
    draws <- m$predict(x_new, inp$nsamp, x_mean_new)
    dens <- m$eval_pred_dens(D(inp$y_eval), x_new[1:1, , drop = FALSE], inp$nsamp,
                             if (is.null(x_mean_new)) NULL else x_mean_new[1:1, , drop = FALSE],
                             log = TRUE)
    list(means = moments$pred_mean, vars = moments$pred_var, predict = draws,
         eval_pred_dens = dens)
  },

  train = function(inp) {
    # Feed fixed base samples to R's own training loop: a GPR_class whose
    # gen_batch replays them, passed to shrinkGPR() as a continuation model
    replay <- set_replay(lapply(inp$zs, D))
    m <- new_gpr(inp, GPR_replay)
    load_state(m, inp$state, dtype)
    opt <- optim_adam(m$parameters, lr = 1e-3, weight_decay = 1e-3)
    cont <- structure(list(model = m, loss = Inf, last_model = m, optimizer = opt,
                           model_internals = list()), class = "shrinkGPR")
    df <- data.frame(y = as.numeric(as_array(D(inp$y))), as_array(D(inp$x)))
    res <- shrinkGPR(y ~ ., data = df, n_epochs = length(inp$zs), auto_stop = FALSE,
                     display_progress = FALSE, cont_model = cont, device = cpu)
    c(list(loss_stor = res$loss_stor, n_replayed = replay$i), state_of(res$last_model))
  },

  end_to_end = function(inp) {
    # x_mean = [1, second covariate] on the Python side, i.e. formula_mean = ~ X2
    df <- data.frame(y = as.numeric(as_array(D(inp$y))), as_array(D(inp$x)))
    df_new <- data.frame(y = as.numeric(as_array(D(inp$y_new))), as_array(D(inp$x_new)))
    torch_manual_seed(inp$seed)
    if (isTRUE(inp$with_mean)) {
      res <- shrinkGPR(y ~ ., data = df, formula_mean = ~ X2, n_epochs = inp$n_epochs,
                       auto_stop = FALSE, display_progress = FALSE, device = cpu)
    } else {
      res <- shrinkGPR(y ~ ., data = df, n_epochs = inp$n_epochs, auto_stop = FALSE,
                       display_progress = FALSE, device = cpu)
    }
    torch_manual_seed(11)
    moments <- calc_pred_moments(res, newdata = df_new, nsamp = 20)
    torch_manual_seed(12)
    draws <- predict(res, newdata = df_new, nsamp = 20)
    torch_manual_seed(13)
    lpds <- LPDS(res, df_new[1, ], nsamp = 20)
    list(loss_stor = res$loss_stor, loss = res$loss, means = moments$means,
         vars = moments$vars, predict = draws, lpds = lpds)
  }
)

results <- lapply(input$cases, function(case) {
  out <- tryCatch(handlers[[case$fn]](case$inputs),
                  error = function(e) list(error = conditionMessage(e)))
  enc(out)
})
names(results) <- vapply(input$cases, function(case) case$name, "")
write_output(results, args[2])
