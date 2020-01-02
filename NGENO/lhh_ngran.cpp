#include <torch/extension.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <cmath>
#include "thread_rand.h"

double normal_estimate(
	at::Tensor Pts_Patch,
	at::Tensor probabilities,
    at::Tensor out_N,
	at::Tensor out_gradients,
	int randSeed,
	int hypCount,
	int Score_type,
	float inlier_thresh,
	float inlier_beta
)
{
	int cMin = 2;
    double mu = 0, sigma = 0.01;
	at::TensorAccessor<float, 3> PtsAccess = Pts_Patch.accessor<float, 3>();
	at::TensorAccessor<float, 3> pAccess = probabilities.accessor<float, 3>();
	at::TensorAccessor<float, 1> NAccess = out_N.accessor<float, 1>();
	at::TensorAccessor<float, 3> gAccess = out_gradients.accessor<float, 3>();
	std::vector<float> Score(hypCount);
	std::vector<float> wPts;
	std::vector<std::vector<float>> NormHy(hypCount);
	std::vector<float> OriginPts (3)
	OriginPts[0] = 0
	OriginPts[1] = 0
	OriginPts[2] = 0
	int nPts = PtsAccess.size(1);
	std::vector<std::vector<float>> Pts(nPts);
	for (int c = 0; c < nPts ; c++)
	{
	    Pts[c] = std::vector<float>(3);
	    for (int d = 0; d < 3; d++){
	         Pts[c][d] = PtsAccess[d][c][0];
	    }

		wPts.push_back(pAccess[0][c][0]);
	}

	ThreadRand::init(randSeed);
	std::discrete_distribution<int> multinomialDist(wPts.begin(), wPts.end());
	std::vector<std::vector<int>> minSets(hypCount);
#pragma omp parallel for
	for (int h = 0; h < hypCount; h++)
	{
		unsigned threadID = omp_get_thread_num();
		minSets[h] = std::vector<int>(cMin);
		for (int j = 0; j < cMin; j++)
		{
			// choose a correspondence based on the provided weights/probabilities
			int cIdx = multinomialDist(ThreadRand::generators[threadID]);

			minSets[h][j] = cIdx;

		}

	}

	// compute the score for the hyp (choose the highest score to compute the loss)
	for (int h = 0; h < hypCount; h++) {
		std::vector<std::vector<float>> Hpts(3);
		std::vector<float> Vn(3);
		std::vector<float> dist_Pts2Plane(nPts);
		std::vector<float> score_PerPts(nPts);
		double ScoreSum = 0;
		for (int j = 0; j < cMin; j++) {
			Hpts[j] = Pts[minSets[h][j]];
		}
		Hpts[2] = OriginPts

		Vn = { 0, 0, 0 };
		double norm2 = 0;
		double Vec1[3];
		double Vec2[3];
		for (int i = 0; i < 3; i++) {
			Vec1[i] = Hpts[0][i] - Hpts[1][i];
			Vec2[i] = Hpts[0][i] - Hpts[2][i];
		}
		Vn[0] = Vec1[1] * Vec2[2] - Vec1[2] * Vec2[1];
		Vn[1] = Vec1[2] * Vec2[0] - Vec1[0] * Vec2[2];
		Vn[2] = Vec1[0] * Vec2[1] - Vec1[1] * Vec2[0];
		for (int i = 0; i < 3; i++) {
			norm2 += Vn[i] * Vn[i];
		}
		norm2 = std::sqrt(norm2);
		Score[h] = 0;
		if (norm2 < 0.0001) {
			Score[h] = -1;
			continue;
		}
		else {
			for (int k = 0; k < 3; k++) {
				Vn[k] /= norm2;
			}
			NormHy[h] = Vn;
			if (Score_type) {
				for (int i = 0; i < nPts; i++) {
					dist_Pts2Plane[i] = 0;
					//dist_Pts2Plane[i] = comput_dist(Pts[i], Hpts[1], Vn, 3);
					for (int j = 0; j < 3; j++) {
						dist_Pts2Plane[i] += (Pts[i][j] - Hpts[1][j])*Vn[j];
					}
					dist_Pts2Plane[i] = abs(dist_Pts2Plane[i]);

					score_PerPts[i] = inlier_beta * (inlier_thresh - dist_Pts2Plane[i]);
					score_PerPts[i] = 1 / (exp(-score_PerPts[i]) + 1);
					ScoreSum += score_PerPts[i];
				}
			}
			else {
				for (int i = 0; i < nPts; i++) {
					dist_Pts2Plane[i] = 0;

					for (int j = 0; j < 3; j++) {
						dist_Pts2Plane[i] += (Pts[i][j] - Hpts[1][j])*Vn[j];
					}
					dist_Pts2Plane[i] = abs(dist_Pts2Plane[i]);

                    score_PerPts[i] = exp(-(dist_Pts2Plane[i] - mu)*(dist_Pts2Plane[i] - mu) / (2 * sigma));
					ScoreSum += score_PerPts[i];
				}

			}
			Score[h] = ScoreSum;
		}
	}

	double bestScore = -1.0;
	int bestH;
	for (int h = 0; h < hypCount; h++) {
		if (bestScore < Score[h]) {
			bestScore = Score[h];
			bestH = h;
		}

		for (int c = 0; c < minSets[h].size(); c++) {
			int cIdx = minSets[h][c];
			gAccess[0][cIdx][0] += 1;
		}
	}
	for (int i =0; i < 3; i++){
	NAccess[i] = NormHy[bestH][i];
	}


	return bestScore;
}


// register C++ functions for use in Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("normal_estimate", &normal_estimate, "Computes normal from given Pts");
}
