# gender-prediction-MLOps
# Validation des cut-offs du score non financier (étapes 2 à 8).
# Entrée : une ligne par dossier.
# Convention : un score égal à une borne tombe dans la tranche supérieure (comme dans Tableau).
# Sortie : résumé dans la console + validation_cuts.xlsx (un onglet par contrôle).
# Packages : install.packages(c("dplyr", "tidyr", "purrr", "lubridate", "writexl"))

suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(purrr); library(lubridate); library(writexl)
})

# ------------------------------------------------------------------ paramètres
COL_SCORE  <- "score"
COL_TARGET <- "top_ctx24m"     # 0/1
COL_SAMPLE <- "sample"         # "mod" / "ht"
COL_DATE   <- "dat_mep_cgi"
COL_SRI    <- NULL             # ex. "sri_niveau_de_risque" pour la matrice croisée
VARIABLES  <- character(0)     # variables du score (déjà en modalités) pour PSI/IV par année

CUTS   <- c(277, 298, 321, 341)
LABELS <- c("---", "-", "+", "++", "+++")
SEUILS <- list(min_vol = 0.05, min_defauts = 30, min_ratio = 1.5, max_dr_gap = 0.30, max_psi = 0.10)

ANNEE_FOCUS <- 2024; TRANCHE_FOCUS <- "+"; POINT_FOCUS <- 297

# ------------------------------------------------------------------ outils
psi <- function(p, q, eps = 1e-6) {
  p <- pmax(as.numeric(p), eps); q <- pmax(as.numeric(q), eps)
  sum((q - p) * log(q / p))
}

chr <- function(x) ifelse(is.na(x), "(manquant)", as.character(x))

iv <- function(x, y) {
  t <- table(chr(x), factor(y, levels = c(0, 1)))
  if (sum(t[, 1]) == 0 || sum(t[, 2]) == 0) return(NA_real_)
  g <- (t[, 1] + 0.5) / sum(t[, 1] + 0.5)
  b <- (t[, 2] + 0.5) / sum(t[, 2] + 0.5)
  sum((g - b) * log(g / b))
}

add_tranche <- function(d, cuts) {
  d$tranche <- cut(d[[COL_SCORE]], c(-Inf, cuts, Inf), labels = LABELS, right = FALSE)
  d
}

band_table <- function(d, by = NULL) {
  d %>%
    group_by(across(all_of(c(by, "tranche"))), .drop = FALSE) %>%
    summarise(n = n(), defauts = sum(.data[[COL_TARGET]]), .groups = "drop") %>%
    group_by(across(all_of(by))) %>%
    mutate(dr = defauts / n, vol = n / sum(n)) %>%
    ungroup()
}

check <- function(t) {
  t <- t[match(LABELS, as.character(t$tranche)), ]
  t$ratio_vs_suivante <- t$dr / lead(t$dr)
  v <- list(
    monotone    = isTRUE(all(diff(t$dr) < 0)),
    ratio_min   = min(t$ratio_vs_suivante, na.rm = TRUE),
    vol_min     = min(t$vol),
    defauts_min = as.integer(min(t$defauts))
  )
  v$OK <- v$monotone && v$ratio_min >= SEUILS$min_ratio &&
    v$vol_min >= SEUILS$min_vol && v$defauts_min >= SEUILS$min_defauts
  list(t = t, v = v)
}

inversions <- function(t, per) {
  t %>%
    arrange(.data[[per]], tranche) %>%
    group_by(.data[[per]]) %>%
    mutate(tranche_suiv = lead(as.character(tranche)),
           dr_suiv = lead(dr), defauts_suiv = lead(defauts)) %>%
    ungroup() %>%
    filter(!is.na(dr_suiv), dr <= dr_suiv) %>%
    transmute(periode = .data[[per]], paire = paste(tranche, "<=", tranche_suiv),
              dr, dr_suiv, defauts, defauts_suiv)
}

points_score <- function(d, by = NULL) {
  d %>%
    group_by(across(all_of(c(by, COL_SCORE, "tranche")))) %>%
    summarise(n = n(), defauts = sum(.data[[COL_TARGET]]), .groups = "drop") %>%
    group_by(across(all_of(by))) %>%
    mutate(dr = defauts / n, vol = n / sum(n)) %>%
    ungroup()
}

sensibilite <- function(d, cuts, deltas = -5:5) {
  out <- list()
  for (i in seq_along(cuts)) for (dl in deltas) {
    cc <- cuts; cc[i] <- cc[i] + dl
    if (is.unsorted(cc, strictly = TRUE)) next
    dd <- add_tranche(d, cc)
    row <- list(borne = i, valeur = cc[i], delta = dl); vols <- list()
    for (s in c("mod", "ht")) {
      ck <- check(band_table(dd[dd[[COL_SAMPLE]] == s, ]))
      row[[paste0("monotone_", s)]]  <- ck$v$monotone
      row[[paste0("ratio_min_", s)]] <- ck$v$ratio_min
      row[[paste0("vol_min_", s)]]   <- ck$v$vol_min
      vols[[s]] <- ck$t$vol
    }
    row$psi_mod_ht <- psi(vols$mod, vols$ht)
    out[[length(out) + 1]] <- as_tibble(row)
  }
  bind_rows(out)
}

stab_variables <- function(d, variables) {
  ref <- d[d[[COL_SAMPLE]] == "mod", ]
  map_dfr(variables, function(v) {
    p_ref <- prop.table(table(chr(ref[[v]])))
    map_dfr(sort(unique(d$annee)), function(a) {
      g <- d[d$annee == a, ]
      p <- prop.table(table(chr(g[[v]])))
      idx <- union(names(p_ref), names(p))
      pr <- as.numeric(p_ref[idx]); pr[is.na(pr)] <- 0
      pa <- as.numeric(p[idx]);     pa[is.na(pa)] <- 0
      tibble(variable = v, annee = a, psi_vs_mod = psi(pr, pa),
             iv = iv(g[[v]], g[[COL_TARGET]]))
    })
  })
}

# ------------------------------------------------------------------ validation
run <- function(df, cuts = CUTS, out = "validation_cuts.xlsx") {
  dt <- as.Date(df[[COL_DATE]])
  df$annee <- year(dt)
  df$trimestre <- paste0(year(dt), " T", quarter(dt))
  nuls <- df[is.na(df[[COL_SCORE]]), ]
  d <- add_tranche(df[!is.na(df[[COL_SCORE]]), ], cuts)
  res <- list()

  # Étapes 2-3 : mod, ht, comparaison
  g <- band_table(d, by = COL_SAMPLE)
  tabs <- list()
  for (s in c("mod", "ht")) {
    ck <- check(g[g[[COL_SAMPLE]] == s, ])
    tabs[[s]] <- ck$t
    res[[paste0("tranches_", s)]] <- ck$t
    cat(sprintf("[%s] %s | monotone=%s ratio_min=%.2f vol_min=%.1f%% defauts_min=%d\n",
                s, ifelse(ck$v$OK, "OK", "KO"), ck$v$monotone, ck$v$ratio_min,
                100 * ck$v$vol_min, ck$v$defauts_min))
  }
  comp <- tibble(tranche = LABELS,
                 vol_mod = tabs$mod$vol, vol_ht = tabs$ht$vol,
                 dr_mod = tabs$mod$dr, dr_ht = tabs$ht$dr) %>%
    mutate(ecart_dr_rel = abs(dr_ht - dr_mod) / dr_mod)
  p_mh <- psi(comp$vol_mod, comp$vol_ht)
  res$mod_vs_ht <- comp
  cat(sprintf("[mod vs ht] PSI=%.4f | écart DR max=%.1f%% -> %s\n", p_mh,
              100 * max(comp$ecart_dr_rel),
              ifelse(p_mh <= SEUILS$max_psi && max(comp$ecart_dr_rel) <= SEUILS$max_dr_gap, "OK", "KO")))

  # Étape 4 : stabilité par année / trimestre
  vol_ref <- setNames(comp$vol_mod, LABELS)
  for (per in c("annee", "trimestre")) {
    t <- band_table(d, by = per)
    stab <- t %>%
      group_by(.data[[per]]) %>%
      summarise(psi_vs_mod = psi(vol_ref[as.character(tranche)], vol),
                n_dossiers = sum(n), n_defauts = sum(defauts), .groups = "drop")
    res[[paste0("vol_", per)]] <- t %>% select(all_of(per), tranche, vol) %>%
      pivot_wider(names_from = tranche, values_from = vol)
    res[[paste0("dr_", per)]] <- t %>% select(all_of(per), tranche, dr) %>%
      pivot_wider(names_from = tranche, values_from = dr)
    res[[paste0("stab_", per)]] <- stab
    res[[paste0("inversions_", per)]] <- inversions(t, per)
    cat(sprintf("[%s] PSI max=%.3f | inversions=%d\n", per,
                max(stab$psi_vs_mod), nrow(res[[paste0("inversions_", per)]])))
  }

  # Étape 5 : points de score atypiques (|z| > 2 et vol >= 1 %)
  pts <- points_score(d, by = COL_SAMPLE) %>%
    left_join(g %>% select(all_of(COL_SAMPLE), tranche, dr_tranche = dr),
              by = c(COL_SAMPLE, "tranche")) %>%
    mutate(z = (dr - dr_tranche) / sqrt(dr_tranche * (1 - dr_tranche) / n),
           atypique = abs(z) > 2 & vol >= 0.01)
  res$points_score <- pts
  cat("[points atypiques]\n"); print(as.data.frame(filter(pts, atypique)), digits = 3)
  cat(sprintf("[point %s]\n", POINT_FOCUS))
  print(as.data.frame(pts[pts[[COL_SCORE]] == POINT_FOCUS, ]), digits = 3)

  # Étape 6 : scores nuls
  res$scores_nuls <- nuls %>%
    group_by(.data[[COL_SAMPLE]]) %>%
    summarise(n = n(), defauts = sum(.data[[COL_TARGET]]), dr = defauts / n, .groups = "drop")
  cat("[scores nuls]\n"); print(as.data.frame(res$scores_nuls), digits = 3)

  # Étape 7 : sensibilité des bornes (±5 points)
  res$sensibilite <- sensibilite(d, cuts)

  # Étape 8 : matrice croisée avec le SRI
  if (!is.null(COL_SRI)) {
    res$matrice_dr <- d %>%
      group_by(.data[[COL_SRI]], tranche) %>%
      summarise(dr = mean(.data[[COL_TARGET]]), .groups = "drop") %>%
      pivot_wider(names_from = tranche, values_from = dr)
    res$matrice_vol <- d %>%
      count(.data[[COL_SRI]], tranche) %>%
      mutate(vol = n / sum(n)) %>% select(-n) %>%
      pivot_wider(names_from = tranche, values_from = vol)
  }

  # Focus 2024 / tranche "+"
  f <- d %>% filter(annee == ANNEE_FOCUS, tranche == TRANCHE_FOCUS)
  res$focus_annee_tranche <- points_score(f)
  cat(sprintf("[%s / '%s'] n=%d, défauts=%d\n", ANNEE_FOCUS, TRANCHE_FOCUS,
              nrow(f), as.integer(sum(f[[COL_TARGET]]))))
  print(table(f[[COL_SAMPLE]]))

  # Stabilité des variables (PSI / IV par année)
  if (length(VARIABLES) > 0) res$variables <- stab_variables(d, VARIABLES)

  names(res) <- substr(names(res), 1, 31)
  write_xlsx(lapply(res, as.data.frame), out)
  cat("->", out, "\n")
  invisible(res)
}

# ------------------------------------------------------------------ lancement
# df  <- readRDS("score_non_financier.rds")   # à adapter
# res <- run(df)
