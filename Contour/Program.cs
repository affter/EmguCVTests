using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Drawing;

namespace Contour
{
    class Program
    {
        static void Main(string[] args)
        {
            var img = CvInvoke.Imread("../../../source/human.png", ImreadModes.Grayscale);
            UMat edges = new UMat();
            UMat edges2 = new UMat();
            VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
            UMat newEdges = new UMat();
            CvInvoke.Canny(img, edges, 20, 100, 3);
            CvInvoke.Imshow("edges", edges);
            CvInvoke.Dilate(edges, edges, new Mat(), new Point(-1, -1), 1, BorderType.Default, new MCvScalar(255, 0, 0));
            CvInvoke.Erode(edges, edges, new Mat(), new Point(-1, -1), 1, BorderType.Default, new MCvScalar(255, 0, 0));
            CvInvoke.FindContours(edges, contours, newEdges, RetrType.Tree, ChainApproxMethod.ChainApproxSimple);
            CvInvoke.Canny(img, newEdges, 20, 180, 3);
            CvInvoke.AbsDiff(edges, newEdges, edges2);
            CvInvoke.AbsDiff(edges, edges2, edges2);
            CvInvoke.DrawContours(img, contours, -1, new MCvScalar(255, 0, 0));
            CvInvoke.Imshow("newEdges", edges2);
            CvInvoke.Imshow("edges", edges);
            CvInvoke.Imshow("grayscale", img);
            CvInvoke.WaitKey();
        }
    }
}
