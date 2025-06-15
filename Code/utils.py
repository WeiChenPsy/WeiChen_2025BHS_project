import os

def get_paths(subject: str, run: int, base_dir: str = "raw_data/ds004078/derivatives"):
    run_str = str(run)

    annotations_dir = os.path.join(base_dir, "annotations", "time_align", "word-level")
    preproc_dir = os.path.join(base_dir, "preprocessed_data", subject, "MNI")

    func_file = os.path.join(preproc_dir, f"{subject}_task-RDR_run-{run_str}_bold.nii.gz")
    mat_file = os.path.join(annotations_dir, f"story_{run_str}_word_time.mat")

    return func_file, mat_file




import os
from nilearn.plotting import plot_stat_map
from nilearn.glm.contrasts import compute_contrast

def save_zmap_and_betamap(labels, results, contrast_vec, masker, subject, output_dir="output/1stlvl"):
    os.makedirs(output_dir, exist_ok=True)

    # Z-map
    z_data = compute_contrast(labels, results, contrast_vec, stat_type='t').z_score()
    z_img = masker.inverse_transform(z_data)
    z_img_path = os.path.join(output_dir, f"{subject}_zmap.nii.gz")
    z_img.to_filename(z_img_path)

    # β-map
    beta_data = compute_contrast(labels, results, contrast_vec, stat_type='t').effect_size()
    beta_img = masker.inverse_transform(beta_data)
    beta_img_path = os.path.join(output_dir, f"{subject}_beta.nii.gz")
    beta_img.to_filename(beta_img_path)


    print(f" Z-map: {z_img_path}")
    print(f" β-map: {beta_img_path}")






import os
from nilearn import plotting
from nilearn.glm import threshold_stats_img

def save_zmap_figures(z_map_img, anat_img, subject, output_dir="output/1stlvl/figure",
                      threshold=3.1, alpha=0.001, cluster_threshold=50):
    os.makedirs(output_dir, exist_ok=True)

    # 儲存未校正 z-map 圖
    display1 = plotting.plot_stat_map(
        z_map_img,
        bg_img=anat_img,
        threshold=threshold,
        display_mode='z',
        cut_coords=[-40, -25, -10, 0, 10, 25, 40],
        black_bg=True,
        title=f"{subject} Speech > Baseline (uncorrected)"
    )
    uncorrected_path = os.path.join(output_dir, f"{subject}_zmap_uncorrected.png")
    display1.savefig(uncorrected_path)
    display1.close()

    # 計算經校正的 z-map
    z_map_thresh, _ = threshold_stats_img(
        z_map_img,
        height_control='fdr',
        alpha=alpha,
        cluster_threshold=cluster_threshold
    )

    # 儲存校正後 z-map 圖
    display2 = plotting.plot_stat_map(
        z_map_thresh,
        bg_img=anat_img,
        threshold=threshold,
        display_mode='z',
        cut_coords=[-40, -25, -10, 0, 10, 25, 40],
        black_bg=True,
        title=f"{subject} Speech > Baseline (FDR α={alpha}, cluster>{cluster_threshold})"
    )
    thresholded_path = os.path.join(output_dir, f"{subject}_zmap_thresholded.png")
    display2.savefig(thresholded_path)
    display2.close()

    print(f"saved：\n{uncorrected_path}\n{thresholded_path}")