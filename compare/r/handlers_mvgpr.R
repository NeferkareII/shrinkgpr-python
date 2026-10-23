# Handlers for the multivariate models (cases in compare/cases_mvgpr.py and
# compare/cases_mvtpr.py). Sourced by numerics.R. The model handlers take the
# class name ("MVGPR" or "MVTPR") as inp$model.

mv_classes <- list(MVGPR = shrinkGPR:::MVGPR_class, MVTPR = shrinkGPR:::MVTPR_class)
mv_r_class <- list(MVGPR = "shrinkMVGPR", MVTPR = c("shrinkMVGPR", "shrinkMVTPR"))
mv_fit_fun <- list(MVGPR = shrinkMVGPR, MVTPR = shrinkMVTPR)

# A subclass of an MV model class whose gen_batch replays the list set by set_replay()
mv_replay_class <- function(model) {
  nn_module(
    inherit = mv_classes[[model]],
    gen_batch = function(n_latent) {
      r <- get(".replay", envir = globalenv())
      r$env$i <- r$env$i + 1
      r$zs[[r$env$i]]
    }
  )
}

new_mv <- function(inp, module_class = mv_classes[[inp$model]]) {
  args <- list(D(inp$y), D(inp$x), a = inp$a, c = inp$c, eta = inp$eta,
               a_Om = inp$a_Om, c_Om = inp$c_Om, sigma2_rate = inp$sigma2_rate)
  if (inp$model == "MVTPR") args <- c(args, list(nu_alpha = inp$nu_alpha, nu_beta = inp$nu_beta))
  args <- c(args, list(n_layers = inp$n_layers, flow_func = flows[[inp$flow]],
                       flow_args = inp$flow_args, kernel_func = kernels[[inp$kernel]],
                       device = cpu))
  m <- do.call(module_class, args)
  # The prior buffers are created as float32; cast everything to the precision
  m$to(dtype = dtype)
  m
}

# A model with response y (N x M), only used to call its methods
mv_dummy <- function(model, y) {
  m <- mv_classes[[model]](y, torch_randn(y$size(1), 1), n_layers = 1, flow_func = sylvester,
                           flow_args = list(), device = cpu)
  m$to(dtype = dtype)
  m
}

# Likelihood, prior and Jacobian terms, copied from MVGPR_class$elbo and
# MVTPR_class$elbo (R/MVGPR_class.R, R/MVTPR_class.R)
mv_elbo_parts <- function(m, zk_pos, model) {
  M <- m$M
  omega_comp <- M * (M - 1) / 2 + M - 1
  D_uncons <- zk_pos[, 1:(M * (M - 1) / 2)]
  S_uncons <- zk_pos[, (M * (M - 1) / 2 + 1):omega_comp]
  theta_zk <- zk_pos[, (omega_comp + 1):(omega_comp + m$d)]
  tau_zk <- zk_pos[, (omega_comp + m$d + 1)]
  tau_Om_zk <- zk_pos[, (omega_comp + m$d + 2)]
  sigma_zk <- zk_pos[, (omega_comp + m$d + 3)]

  theta_zk <- shrinkGPR:::res_protector_autograd(theta_zk)
  tau_zk <- shrinkGPR:::res_protector_autograd(tau_zk, tol = 1e-4)
  sigma_zk <- shrinkGPR:::res_protector_autograd(sigma_zk, tol = 1e-4)

  K <- m$kernel_func(theta_zk, tau_zk, m$x)
  D_chol_zk <- m$make_corr_chol(D_uncons)

  b <- 4
  S_uncons_c <- b * torch_tanh(S_uncons / b)
  S_M <- -torch_sum(S_uncons_c, dim = 2, keepdim = TRUE)
  S_diag <- torch_exp(torch_cat(list(S_uncons_c, S_M), dim = 2))
  S <- torch_diag_embed(S_diag)
  log_det_S <- torch_sum(S_uncons_c, dim = 2) +
    torch_sum(torch_log(torch_clamp(1 / torch_cosh(S_uncons / b)$pow(2), min = 1e-12)), dim = 2)

  L_Om <- torch_bmm(S, D_chol_zk$L)
  eps_diag <- 0.03
  diag <- torch_diagonal(L_Om, dim1 = 2, dim2 = 3)
  diag2 <- eps_diag + nnf_softplus(diag - eps_diag, beta = 10)
  L_Om2 <- L_Om$clone()
  L_Om2$diagonal(dim1 = 2, dim2 = 3)$copy_(diag2)

  if (model == "MVTPR") {
    nu_raw <- zk_pos[, (omega_comp + m$d + 4)]
    nu_zk <- nu_raw + 2
    likelihood <- m$ldt(K, L_Om2, sigma_zk, nu_zk)$mean()
  } else {
    likelihood <- m$ldnorm(K, L_Om2, sigma_zk)$mean()
  }

  diag_LD <- torch_diagonal(D_chol_zk$L, dim1 = 2, dim2 = 3)
  lkj_w <- torch_tensor(as.numeric(m$M - seq_len(m$M)), device = diag_LD$device)
  lkj_term <- torch_sum((lkj_w + 2 * (m$prior_eta - 1)) * torch_log(diag_LD), dim = 2)
  prior <- m$ltg(theta_zk, m$prior_a, m$prior_c, tau_zk)$sum(dim = 2)$mean() +
    m$ldf(tau_zk, 2 * m$prior_c, 2 * m$prior_a)$mean() +
    lkj_term$mean() +
    m$ltg(S_diag, m$prior_a_Om, m$prior_c_Om, tau_Om_zk)$sum(dim = 2)$mean() +
    m$ldf(tau_Om_zk, 2 * m$prior_c_Om, 2 * m$prior_a_Om)$mean() +
    m$lexp(sigma_zk, m$prior_rate)$mean()
  if (model == "MVTPR") prior <- prior + m$ldg(nu_raw, m$nu_alpha, m$nu_beta)$mean()

  diag_logJ <- torch_sum(torch_log(torch_sigmoid(10.0 * (diag - eps_diag))), dim = 2)
  list(likelihood = likelihood, prior = prior, logJ_D = D_chol_zk$logJ$mean(),
       log_det_S = log_det_S$mean(), diag_logJ = diag_logJ$mean())
}

# R formula interface for an N x M response and N x p covariates
mv_formula_data <- function(y, x) {
  y <- as_array(y); x <- as_array(x)
  colnames(y) <- paste0("y", seq_len(ncol(y)))
  colnames(x) <- paste0("X", seq_len(ncol(x)))
  f <- as.formula(paste0("cbind(", paste(colnames(y), collapse = ", "), ") ~ ."))
  list(formula = f, data = data.frame(y, x))
}

handlers$mv_corr_chol <- function(inp) {
  m <- mv_dummy(inp$model, torch_randn(4, inp$M))
  out <- m$make_corr_chol(D(inp$u))
  list(L = out$L, logJ = out$logJ)
}

handlers$mv_ldnorm <- function(inp) {
  y <- D(inp$y)
  list(ll = jit$ldnorm_multi(D(inp$L_K), D(inp$L_Om), y, as.integer(y$size(2)),
                             as.integer(y$size(1))))
}

handlers$mv_model <- function(inp) {
  m <- new_mv(inp)
  load_state(m, inp$state, dtype)
  out <- m(D(inp$z))
  elbo <- m$elbo(out$zk, out$log_det_J)
  (-elbo)$backward()
  c(list(zk = out$zk, log_det_J = out$log_det_J, elbo = elbo), grads_of(m))
}

handlers$mv_parts <- function(inp) {
  m <- new_mv(inp)
  load_state(m, inp$state, dtype)
  with_no_grad(mv_elbo_parts(m, D(inp$zk), inp$model))
}

handlers$mv_pred <- function(inp) {
  set_replay(rep(list(D(inp$z)), 4))
  m <- new_mv(inp, mv_replay_class(inp$model))
  load_state(m, inp$state, dtype)
  x_new <- D(inp$x_new)
  moments <- m$calc_pred_moments(x_new, inp$nsamp)
  dens <- m$eval_pred_dens(D(inp$y_eval), x_new[1:1, , drop = FALSE], inp$nsamp, log = TRUE)
  out <- list(means = moments$pred_mean, K = moments$K, Omega = moments$Omega,
              eval_pred_dens = dens)
  if (!is.null(moments$nu)) out$nu <- moments$nu
  if (isTRUE(inp$with_predict)) {
    torch_manual_seed(inp$seed)
    out$predict <- m$predict(x_new, inp$nsamp)
  }
  out
}

handlers$mv_posterior <- function(inp) {
  set_replay(list(D(inp$z)))
  m <- new_mv(inp, mv_replay_class(inp$model))
  load_state(m, inp$state, dtype)
  p <- D(inp$x)$size(2)
  M <- D(inp$y)$size(2)
  fo <- reformulate(paste0("X", seq_len(p)),
                    str2lang(paste0("cbind(", paste0("Y", seq_len(M), collapse = ", "), ")")))
  mod <- structure(list(model = m, model_internals = list(d_cov = p, M = M, terms = terms(fo))),
                   class = mv_r_class[[inp$model]])
  flat_posterior(gen_posterior_samples(mod, nsamp = inp$nsamp))
}

handlers$mv_train <- function(inp) {
  # Feed fixed base samples to R's own training loop: a model whose gen_batch
  # replays them, passed to shrinkMVGPR()/shrinkMVTPR() as a continuation model
  replay <- set_replay(lapply(inp$zs, D))
  m <- new_mv(inp, mv_replay_class(inp$model))
  load_state(m, inp$state, dtype)
  opt <- optim_adam(m$parameters, lr = 1e-4, weight_decay = 1e-3)
  cont <- structure(list(model = m, loss = Inf, last_model = m, optimizer = opt,
                         model_internals = list()), class = mv_r_class[[inp$model]])
  fd <- mv_formula_data(D(inp$y), D(inp$x))
  res <- mv_fit_fun[[inp$model]](fd$formula, data = fd$data, n_epochs = length(inp$zs),
                                 auto_stop = FALSE, display_progress = FALSE,
                                 cont_model = cont, device = cpu)
  c(list(loss_stor = res$loss_stor, n_replayed = replay$i), state_of(res$last_model))
}

handlers$mv_end_to_end <- function(inp) {
  fd <- mv_formula_data(D(inp$y), D(inp$x))
  df_new <- data.frame(as_array(D(inp$y_new)), as_array(D(inp$x_new)))
  names(df_new) <- names(fd$data)
  args <- list(fd$formula, data = fd$data, n_epochs = inp$n_epochs, auto_stop = FALSE,
               display_progress = FALSE, device = cpu)
  if (inp$model == "MVTPR") args <- c(args, list(nu_alpha = inp$nu_alpha, nu_beta = inp$nu_beta))
  torch_manual_seed(inp$seed)
  res <- do.call(mv_fit_fun[[inp$model]], args)
  torch_manual_seed(11)
  moments <- calc_pred_moments(res, newdata = df_new, nsamp = 20)
  out <- list(loss_stor = res$loss_stor, loss = res$loss, means = moments$means,
              K = moments$K, Omega = moments$Omega)
  if (!is.null(moments$nu)) out$nu <- as.numeric(moments$nu)
  if (isTRUE(inp$with_predict)) {
    torch_manual_seed(12)
    out$predict <- predict(res, newdata = df_new, nsamp = 20)
  }
  torch_manual_seed(13)
  out$lpds <- LPDS(res, df_new[1, ], nsamp = 20)
  torch_manual_seed(15)
  marg <- gen_marginal_samples(res, "X1", nsamp = 4, fixed_x = df_new[1, ], n_eval_points = 6,
                               display_progress = FALSE)
  out$marg <- marg$mean_pred
  out$marg.grid <- marg$grid
  out
}
