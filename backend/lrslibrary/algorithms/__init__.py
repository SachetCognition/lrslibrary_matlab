"""Algorithm package - auto-discovers and imports all algorithm submodules."""


def discover_algorithms() -> None:
    """Import all algorithm submodules to trigger @register decorators.

    Called lazily to avoid circular imports (registry <-> algorithms).
    """
    # RPCA algorithms (47)
    # LRR algorithms (7)
    from lrslibrary.algorithms.lrr import (  # noqa: F401
        fast_ladmap,
        ladmap,
        lrr_adm,
        lrr_alm,
        lrr_ealm,
        lrr_ialm,
        rosl,
    )

    # MC algorithms (16)
    from lrslibrary.algorithms.mc import (  # noqa: F401
        fpc,
        grouse,
        ialm_mc,
        lmafit,
        lmafit_sms,
        lrgeomcg,
        mc_logdet,
        mc_nmf,
        op_rpca,
        optspace,
        or1mp,
        pg_rmc,
        rpca_gd,
        scgrassmc,
        svp,
        svt_mc,
    )

    # NMF algorithms (15)
    from lrslibrary.algorithms.nmf import (  # noqa: F401
        deep_semi_nmf,
        drmf,
        enmf,
        inmf,
        lnmf,
        manh_nmf,
        nenmf,
        nmf_als,
        nmf_als_obs,
        nmf_dtu,
        nmf_ls2,
        nmf_mu,
        nmf_pg,
        pnmf,
        semi_nmf,
    )

    # NTF algorithms (7)
    from lrslibrary.algorithms.ntf import (  # noqa: F401
        bcu_ncp,
        bcu_ntd,
        beta_ntf,
        lra_ntd,
        ntd_apg,
        ntd_hals,
        ntd_mu,
    )
    from lrslibrary.algorithms.rpca import (  # noqa: F401
        adm,
        alm,
        apg,
        apg_partial,
        as_rpca,
        brpca_md,
        brpca_md_nss,
        decolor,
        dual,
        ealm,
        flip_spcp_max_qn,
        flip_spcp_sum_spg,
        fpcp,
        fw_t,
        ga,
        gm,
        godec,
        gregodec,
        ialm,
        ialm_blws,
        ialm_lmsvds,
        l1f,
        lag_spcp_qn,
        lag_spcp_spg,
        lsadm,
        mbrmf,
        mog_rpca,
        noncvx_rpca,
        nsa1,
        nsa2,
        oprmf,
        pcp,
        prmf,
        pspg,
        r2pcp,
        regl1_alm,
        rpca_classic,
        spcp_stub,
        ssgodec,
        stoc_rpca,
        svt,
        tfocs_ec,
        tfocs_ic,
        tga,
        vbrpca,
    )

    # ST algorithms (5)
    from lrslibrary.algorithms.st import (  # noqa: F401
        gosus,
        grasta,
        medrop,
        prost,
        reprocs,
    )

    # TD algorithms (14)
    from lrslibrary.algorithms.td import (  # noqa: F401
        cp2,
        cp_als,
        cp_apr,
        horpca_ialm,
        horpca_s,
        horpca_s_ncx,
        hosvd,
        itl,
        ostd,
        rlrt,
        rstd,
        t_svd,
        tucker_adal,
        tucker_als,
    )

    # TTD algorithms (4)
    from lrslibrary.algorithms.ttd import (  # noqa: F401
        admm_ttd,
        mamr,
        rmamr,
        three_wd,
    )
