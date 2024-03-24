from manim import *

class MatrixDecomposition(Scene):
    def construct(self):
        # Step 1: Generate Rank-Deficient Matrix W
        matrix_w = Matrix([
            ["W"]
        ]).scale(1.5)
        self.play(Write(matrix_w))

        # Step 2: Calculate Rank of W
        rank_text = Tex("Rank of W: 2").next_to(matrix_w, direction=DOWN)
        self.play(Write(rank_text))

        # Step 3: Perform SVD on W
        svd_text = Tex("Performing SVD on W").next_to(rank_text, direction=DOWN)
        self.play(Write(svd_text))

        # Step 4: Compute Reduced Rank Matrices B and A
        decompose_text = Tex("Compute B and A").next_to(svd_text, direction=DOWN)
        self.play(Write(decompose_text))

        # Step 5: Compare Original and Reconstructed Outputs
        compare_text = Tex("Comparing Original and Reconstructed Outputs").next_to(decompose_text, direction=DOWN)
        self.play(Write(compare_text))

        # Step 6: Print Total Parameters
        parameters_text = Tex("Total Parameters").next_to(compare_text, direction=DOWN)
        self.play(Write(parameters_text))

        self.wait(2)

