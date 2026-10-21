# Handlers for the ShrinkTPR cases of compare/cases_tpr.py; sourced by numerics.R.

# A TPR_class whose gen_batch replays the base samples set with set_replay()
TPR_replay <- nn_module(
  inherit = shrinkGPR:::TPR_class,
  gen_batch = function(n_latent) {
    r <- get(".replay", envir = globalenv())
    r$env$i <- r$env$i + 1
    r$zs[[r$env$i]]
  }
)

new_tpr <- function(inp, module_class = shrinkGPR:::TPR_class) {
  m <- module_class(D(inp$y), D(inp$x), D(inp$x_mean),
                    a = inp$a, c = inp$c, a_mean = inp$a_mean, c_mean = inp$c_mean,
                    sigma2_rate = inp$sigma2_rate, nu_alpha = inp$nu_alpha,
                    nu_beta = inp$nu_beta, n_layers = inp$n_layers,
                    flow_func = flows[[inp$flow]], flow_args = inp$flow_args,
                    kernel_func = kernels[[inp$kernel]], device = cpu)
  m$to(dtype = dtype)
  m
}

# TPR_class$predict draws chi-squared variables with stats::rgamma, imported
# into the package namespace. Replace it there by the draws of the Python side
# and record the degrees of freedom it was asked for.
set_rgamma <- function(chi2) {
  draws <- as.numeric(unlist(chi2$data))
  rec <- new.env()
  fake <- function(n, shape, rate = 1, scale = 1 / rate) {
    if (n != length(draws) || !isTRUE(all.equal(rate, 0.5))) stop("unexpected rgamma call")
    rec$df <- 2 * as.numeric(shape)
    draws
  }
  imports <- parent.env(asNamespace("shrinkGPR"))
  unlockBinding("rgamma", imports)
  assign("rgamma", fake, imports)
  lockBinding("rgamma", imports)
  rec
}

tpr_stub <- function() {
  shrinkGPR:::TPR_class(torch_randn(3), torch_randn(3, 1), NULL, n_layers = 1,
                        flow_func = sylvester, flow_args = list(), device = cpu)
}

handlers$tpr_ldg <- function(inp) {
  list(ldg = tpr_stub()$ldg(D(inp$x), S(inp$alpha), S(inp$beta)))
}

handlers$tpr_ldt <- function(inp) {
  list(ll = jit$ldt(K = D(inp$K), L = NULL, sigma2 = D(inp$sigma2), y = D(inp$y),
                    x_mean = D(inp$x_mean), beta = D(inp$beta), nu = D(inp$nu)))
}

handlers$tpr_dt <- function(inp) {
  list(dens = tpr_stub()$dt(D(inp$y), D(inp$mean), D(inp$scale), D(inp$nu)))
}

handlers$tpr <- function(inp) {
  m <- new_tpr(inp)
  load_state(m, inp$state, dtype)
  out <- m(D(inp$z))
  elbo <- m$elbo(out$zk, out$log_det_J)
  (-elbo)$backward()
  c(list(zk = out$zk, log_det_J = out$log_det_J, elbo = elbo), grads_of(m))
}

handlers$tpr_pred <- function(inp) {
  set_replay(rep(list(D(inp$z)), 3))
  rec <- set_rgamma(inp$chi2)
  m <- new_tpr(inp, TPR_replay)
  load_state(m, inp$state, dtype)
  x_new <- D(inp$x_new)
  x_mean_new <- D(inp$x_mean_new)
  moments <- m$calc_pred_moments(x_new, inp$nsamp, x_mean_new)
  torch_manual_seed(inp$seed)
  draws <- m$predict(x_new, inp$nsamp, x_mean_new)
  dens <- m$eval_pred_dens(D(inp$y_eval), x_new[1:1, , drop = FALSE], inp$nsamp,
                           if (is.null(x_mean_new)) NULL else x_mean_new[1:1, , drop = FALSE],
                           log = TRUE)
  list(means = moments$pred_mean, vars = moments$pred_scale, nu = moments$pred_nu,
       predict = draws, chi2_df = rec$df, eval_pred_dens = dens)
}

handlers$tpr_posterior <- function(inp) {
  set_replay(list(D(inp$z)))
  m <- new_tpr(inp, TPR_replay)
  load_state(m, inp$state, dtype)
  p <- D(inp$x)$size(2)
  internals <- list(d_cov = p, terms = terms(reformulate(paste0("X", seq_len(p)), "y")),
                    x_mean = !is.null(inp$x_mean))
  if (internals$x_mean) {
    internals$d_mean <- D(inp$x_mean)$size(2)
    internals$x_mean_names <- paste0("M", seq_len(internals$d_mean))
  }
  mod <- structure(list(model = m, model_internals = internals), class = "shrinkTPR")
  flat_posterior(gen_posterior_samples(mod, nsamp = inp$nsamp))
}

handlers$tpr_train <- function(inp) {
  # As the GPR train handler: replayed base samples through R's own loop
  replay <- set_replay(lapply(inp$zs, D))
  m <- new_tpr(inp, TPR_replay)
  load_state(m, inp$state, dtype)
  opt <- optim_adam(m$parameters, lr = 1e-3, weight_decay = 1e-3)
  cont <- structure(list(model = m, loss = Inf, last_model = m, optimizer = opt,
                         model_internals = list()), class = "shrinkTPR")
  df <- data.frame(y = as.numeric(as_array(D(inp$y))), as_array(D(inp$x)))
  args <- list(y ~ ., data = df, nu_alpha = inp$nu_alpha, nu_beta = inp$nu_beta,
               n_epochs = length(inp$zs), auto_stop = FALSE, display_progress = FALSE,
               cont_model = cont, device = cpu)
  if (!is.null(inp$x_mean)) args$formula_mean <- ~ X2
  res <- do.call(shrinkTPR, args)
  c(list(loss_stor = res$loss_stor, n_replayed = replay$i), state_of(res$last_model))
}

handlers$tpr_end_to_end <- function(inp) {
  # x_mean = [1, second covariate] on the Python side, i.e. formula_mean = ~ X2
  df <- data.frame(y = as.numeric(as_array(D(inp$y))), as_array(D(inp$x)))
  df_new <- data.frame(y = as.numeric(as_array(D(inp$y_new))), as_array(D(inp$x_new)))
  torch_manual_seed(inp$seed)
  if (isTRUE(inp$with_mean)) {
    res <- shrinkTPR(y ~ ., data = df, formula_mean = ~ X2, n_epochs = inp$n_epochs,
                     auto_stop = FALSE, display_progress = FALSE, device = cpu)
  } else {
    res <- shrinkTPR(y ~ ., data = df, n_epochs = inp$n_epochs, auto_stop = FALSE,
                     display_progress = FALSE, device = cpu)
  }
  set_rgamma(inp$chi2)
  torch_manual_seed(11)
  moments <- calc_pred_moments(res, newdata = df_new, nsamp = 20)
  torch_manual_seed(12)
  draws <- predict(res, newdata = df_new, nsamp = 20)
  torch_manual_seed(13)
  lpds <- LPDS(res, df_new[1, ], nsamp = 20)
  list(loss_stor = res$loss_stor, loss = res$loss, means = moments$means,
       vars = moments$vars, nu = moments$nu, predict = draws, lpds = lpds)
}
