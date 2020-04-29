using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;


using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using Emgu.CV.Util;
using Emgu.CV.OCR;
using Emgu.CV.Cvb;
using Emgu.CV.Cuda;
//using Emgu.CV.UI;
using Emgu.CV.VideoStab;


namespace prot1
{
    public partial class Form1 : Form
    {
        Image<Bgr, Byte> imge;
        VideoCapture _capture;
        private Mat _frame;
        private const int Threshold = 1;
        private const int ErodeIterations = 1;
        private const int DilateIterations = 7;
        private static MCvScalar drawingColor = new Bgr(Color.Blue).MCvScalar;

        private async void ProcessFrame(object sender, EventArgs e)
        {
            if (_capture != null && _capture.Ptr != IntPtr.Zero)
            {
                _capture.Retrieve(_frame, 0);
                pictureBox1.Image = _frame.Bitmap;
                double fps = 15;
                await Task.Delay(1000 / Convert.ToInt32(fps));

            }
        }
        public Form1()
        {
            InitializeComponent();
            _capture = new VideoCapture(1);


            _capture.ImageGrabbed += ProcessFrame;
            _frame = new Mat();
            if (_capture != null)
            {
                try
                {
                    _capture.Start();
                }
                catch (Exception ex)
                {
                    MessageBox.Show(ex.Message);
                }
            }
        }

        private async void  button1_Click(object sender, EventArgs e)
        {

            if (_frame.IsEmpty)
            {
                return;
            }

            //try
            //{

                Mat m = new Mat();
                Mat n = new Mat();
                Mat o = new Mat();
                Mat binaryDiffFrame = new Mat();
                Mat denoisedDiffFrame = new Mat();
                Mat finalFrame = new Mat();
                Rectangle cropbox = new Rectangle();


                //
                //OBTENER COLOR ROJO
                //
                imge = _frame.ToImage<Bgr, Byte>();
                //Transformar a espacio de color HSV
                Image<Hsv, Byte> hsvimg = imge.Convert<Hsv, Byte>();

                //extract the hue and value channels
                Image<Gray, Byte>[] channels = hsvimg.Split();  //separar en componentes
                Image<Gray, Byte> imghue = channels[0];            //hsv, channels[0] es hue.
                Image<Gray, Byte> imgval = channels[2];            //hsv, channels[2] es value.

                //Filtro color
                //140 en adelante
                Image<Gray, byte> huefilter = imghue.InRange(new Gray(140), new Gray(255));
                //Filtro colores menos brillantes
                Image<Gray, byte> valfilter = imgval.InRange(new Gray(100), new Gray(255));
                //Filtro de saturación - quitar blancos 
                channels[1]._ThresholdBinary(new Gray(20), new Gray(255)); // Saturacion

                //Unir los filtros para obtener la imagen
                Image<Gray, byte> colordetimg = huefilter.And(valfilter).And(channels[1]);//aqui habia un Not()

                //Colorear imagen
                Image<Bgr, byte> ret = imge;
                var mat = imge.Mat;
                mat.SetTo(new MCvScalar(0, 0, 255), colordetimg);
                mat.CopyTo(ret);
                //Image<Bgr, byte> imgout = ret.CopyBlank();//imagen sin fondo negro
               
                ret._Or(imge);
                //Muestra imagen con los rojos destacados
                pictureBox2.Image = ret.Bitmap;


                //
                //COMIENZA OBTENCION DE BORDES
                //
                CvInvoke.AbsDiff(_frame, ret, n);

                CvInvoke.CvtColor(colordetimg, o, ColorConversion.Bgr2Gray);
                CvInvoke.Threshold(colordetimg, binaryDiffFrame, 5, 255, ThresholdType.Binary);


                CvInvoke.Erode(binaryDiffFrame, denoisedDiffFrame, null, new Point(-1, -1), ErodeIterations, BorderType.Default, new MCvScalar(1));
                CvInvoke.Dilate(denoisedDiffFrame, denoisedDiffFrame, null, new Point(-1, -1), DilateIterations, BorderType.Default, new MCvScalar(1));
                //pictureBox3.Image = denoisedDiffFrame.Bitmap;

                _frame.CopyTo(finalFrame);

                DetectObject(denoisedDiffFrame, finalFrame, cropbox);


                pictureBox3.Image = finalFrame.Bitmap;

                double fps = 15;
                await Task.Delay(1000 / Convert.ToInt32(fps));


           /* }
            catch (Exception ex)
            {
              MessageBox.Show(ex.Message);

             }

                */
        }







        //  
        //COMIENZAN FUNCIONES DE EDDIE
        //
        private void DetectObject(Mat detectionFrame, Mat displayFrame, Rectangle box)
        {
            Image<Bgr, Byte> buffer_im = displayFrame.ToImage<Bgr, Byte>();
            List<Triangle2DF> triangleList = new List<Triangle2DF>();
            List<RotatedRect> boxList = new List<RotatedRect>(); //a box is a rotated rectangle
            
           
          
            //transforma imagen
            //UMat uimage = new UMat();
            // CvInvoke.CvtColor(displayFrame, uimage, ColorConversion.Bgr2Gray);
            using (VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint())
            {
                //VectorOfPoint biggestContour = null;
                IOutputArray hirarchy = null;
                CvInvoke.FindContours(detectionFrame, contours, hirarchy, RetrType.External, ChainApproxMethod.ChainApproxSimple);
                //pictureBox5.Image = detectionFrame.Bitmap;
                CvInvoke.Polylines(detectionFrame, contours, true, new MCvScalar(255, 0, 0), 2, LineType.FourConnected);
                Image<Bgr, Byte> resultadoFinal = displayFrame.ToImage<Bgr, byte>();

                //Circulos
                //double cannyThreshold = 180.0;
                //double circleAccumulatorThreshold = 120;
                //CircleF[] circles = CvInvoke.HoughCircles(detectionFrame, HoughType.Gradient, 2.0, 20.0, cannyThreshold, circleAccumulatorThreshold, 5);

                if (contours.Size > 0)
                {
                    double maxArea = 1000;
                    int chosen = 0;
                    VectorOfPoint contour = null;
                    for (int i = 0; i < contours.Size; i++)
                    {
                        contour = contours[i];
                       
                        double area = CvInvoke.ContourArea(contour);
                        if (area > maxArea)
                        {
                            //  maxArea = area;
                            chosen = i;
                            //}
                            //}
                            
                            //Boxes
                            VectorOfPoint hullPoints = new VectorOfPoint();
                            VectorOfInt hullInt = new VectorOfInt();

                            CvInvoke.ConvexHull(contours[chosen], hullPoints, true);
                            CvInvoke.ConvexHull(contours[chosen], hullInt, false);

                            Mat defects = new Mat();


                            if (hullInt.Size > 3)
                                CvInvoke.ConvexityDefects(contours[chosen], hullInt, defects);

                            box = CvInvoke.BoundingRectangle(hullPoints);
                            CvInvoke.Rectangle(displayFrame, box, drawingColor);//Box rectangulo que encierra el area mas grande
                                                                                // cropbox = crop_color_frame(displayFrame, box);

                            buffer_im.ROI = box;

                            Image<Bgr, Byte> cropped_im = buffer_im.Copy();
                            //pictureBox8.Image = cropped_im.Bitmap;
                            Point center = new Point(box.X + box.Width / 2, box.Y + box.Height / 2);//centro  rectangulo MOUSE
                            Point esquina_superiorI = new Point(box.X, box.Y);
                            Point esquina_superiorD = new Point(box.Right, box.Y);
                            Point esquina_inferiorI = new Point(box.X, box.Y + box.Height);
                            Point esquina_inferiorD = new Point(box.Right, box.Y + box.Height);
                            CvInvoke.Circle(displayFrame, esquina_superiorI, 5, new MCvScalar(0, 0, 255), 2);
                            CvInvoke.Circle(displayFrame, esquina_superiorD, 5, new MCvScalar(0, 0, 255), 2);
                            CvInvoke.Circle(displayFrame, esquina_inferiorI, 5, new MCvScalar(0, 0, 255), 2);
                            CvInvoke.Circle(displayFrame, esquina_inferiorD, 5, new MCvScalar(0, 0, 255), 2);
                            CvInvoke.Circle(displayFrame, center, 5, new MCvScalar(0, 0, 255), 2);
                            VectorOfPoint start_points = new VectorOfPoint();
                            VectorOfPoint far_points = new VectorOfPoint();
                            




                        }
                    }

                    //Dibuja borde amarillo 
                    Image<Bgr, byte> temp = detectionFrame.ToImage<Bgr, byte>();
                    var temp2 = temp.SmoothGaussian(5).Convert<Gray, byte>().ThresholdBinary(new Gray(230), new Gray(255));
                    VectorOfVectorOfPoint contorno = new VectorOfVectorOfPoint();
                    Mat mat = new Mat();
                    CvInvoke.FindContours(temp2, contorno, mat, Emgu.CV.CvEnum.RetrType.External, Emgu.CV.CvEnum.ChainApproxMethod.ChainApproxSimple);
                    for (int i = 0; i < contorno.Size; i++)
                    {
                        
                        VectorOfPoint approxContour = new VectorOfPoint();
                        double perimetro = CvInvoke.ArcLength(contorno[i], true);
                        VectorOfPoint approx = new VectorOfPoint();
                        double area = CvInvoke.ContourArea(contorno[i]);
                        if (area > 1000)
                        {
                            CvInvoke.ApproxPolyDP(contorno[i], approx, 0.04 * perimetro, true);
                            CvInvoke.DrawContours(displayFrame, contorno, i, new MCvScalar(0, 255, 255), 2);
                            //Triangulos y rectangulos
                            //CvInvoke.ApproxPolyDP(contorno[i], approxContour, CvInvoke.ArcLength(contour, true) * 0.05, true);
                            var moments = CvInvoke.Moments(contours[i]);
                            int x = (int)(moments.M10 / moments.M00);
                            int y = (int)(moments.M01 / moments.M00);

                            if (approx.Size == 3) //The contour has 3 vertices, it is a triangle
                            {
                                Point[] pts = approx.ToArray();
                                triangleList.Add(new Triangle2DF(pts[0], pts[1], pts[2]));
                                Triangle2DF triangle = new Triangle2DF(pts[0], pts[1], pts[2]);
                                //Point v0 = new Point(Convert.ToInt32(triangle.V0.X), Convert.ToInt32(triangle.V0.Y));
                                //Point v1 = new Point(Convert.ToInt32(triangle.V1.X), Convert.ToInt32(triangle.V1.Y));
                                //Point v2 = new Point(Convert.ToInt32(triangle.V2.X), Convert.ToInt32(triangle.V2.Y));
                                resultadoFinal.Draw(triangle, new Bgr(Color.Cyan), 1);
                                //CvInvoke.Circle(resultadoFinal, v0, 5, new MCvScalar(0, 255, 0), 2);
                                //CvInvoke.Circle(resultadoFinal, v1, 5, new MCvScalar(0, 255, 0), 2);
                                //CvInvoke.Circle(resultadoFinal, v2, 5, new MCvScalar(0, 255, 0), 2);
                                CvInvoke.DrawContours(resultadoFinal, contorno, i, new MCvScalar(255, 255, 255), 1, LineType.AntiAlias);
                                CvInvoke.PutText(resultadoFinal, "Triangle", new Point(x, y),
                                Emgu.CV.CvEnum.FontFace.HersheySimplex, 0.5, new MCvScalar(0, 255, 255), 2);
                            }
                            if (approx.Size == 4) //The contour has 4 vertices.
                            {
                                boxList.Add(CvInvoke.MinAreaRect(approx));
                                RotatedRect rectangle = CvInvoke.MinAreaRect(approx);
                                CvInvoke.DrawContours(resultadoFinal, contorno, i, new MCvScalar(255, 255, 255), 2, LineType.AntiAlias);
                                resultadoFinal.Draw(rectangle, new Bgr(Color.Cyan), 1);
                                CvInvoke.PutText(resultadoFinal, "Rectangle", new Point(x, y),
                                Emgu.CV.CvEnum.FontFace.HersheySimplex, 0.5, new MCvScalar(0, 255, 255), 2);
                            }
                            if (approx.Size == 5)
                            {
                                //IConvexPolygonF poligono = CvInvoke.MinAreaRect(approx);
                                //resultadoFinal.Draw(poligono, new Bgr(Color.Cyan), 1);
                                CvInvoke.DrawContours(resultadoFinal, contorno, i, new MCvScalar(255, 255, 0), 1, LineType.AntiAlias);
                                CvInvoke.PutText(resultadoFinal, "Pentagon", new Point(x, y),
                                    Emgu.CV.CvEnum.FontFace.HersheySimplex, 0.5, new MCvScalar(0, 255, 255), 2);
                            }
                            if (approx.Size == 6)
                            {
                                //IConvexPolygonF poligono = CvInvoke.MinAreaRect(approx);
                                //resultadoFinal.Draw(poligono, new Bgr(Color.Cyan), 1);
                                CvInvoke.DrawContours(resultadoFinal, contorno, i, new MCvScalar(255, 255, 0), 1, LineType.AntiAlias);
                                CvInvoke.PutText(resultadoFinal, "Oval", new Point(x, y),
                                    Emgu.CV.CvEnum.FontFace.HersheySimplex, 0.5, new MCvScalar(0, 255, 255), 2);
                            }


                            if (approx.Size > 6)
                            {
                                CvInvoke.PutText(resultadoFinal, "Circle", new Point(x, y),
                                    Emgu.CV.CvEnum.FontFace.HersheySimplex, 0.5, new MCvScalar(0, 255, 255), 2);
                                CircleF circle = CvInvoke.MinEnclosingCircle(approx);
                                CvInvoke.DrawContours(resultadoFinal, contorno, i, new MCvScalar(255, 255, 255), 1, LineType.AntiAlias);
                                resultadoFinal.Draw(circle, new Bgr(Color.Cyan), 1);

                            }
                        }

                    }
                   
                    pictureBox5.Image = resultadoFinal.Bitmap;

                }

            }

        }


        private void button2_Click(object sender, EventArgs e)
        {
            if (_frame.IsEmpty)
            {
                return;
            }

            Mat finalFrame = new Mat();
            Mat aux = new Mat();
            Rectangle cropbox = new Rectangle();
            Image<Bgr, byte> img = _frame.ToImage<Bgr, byte>();
            //Transformar a espacio de color HSV
            Image<Hsv, Byte> hsvimg = img.Convert<Hsv, Byte>();

            //extract the hue and value channels
            Image<Gray, Byte>[] channels = hsvimg.Split();  //separar en componentes
            Image<Gray, Byte> imghue = channels[0];            //hsv, channels[0] es hue.
            Image<Gray, Byte> imgval = channels[2];            //hsv, channels[2] es value.

            //Filtro color
            //140 en adelante 
            //funciona  160
            Image<Gray, byte> huefilter = imghue.InRange(new Gray(150), new Gray(255));
            //Filtro colores menos brillantes
            Image<Gray, byte> valfilter = imgval.InRange(new Gray(100), new Gray(255));
            //Filtro de saturación - quitar blancos 
            channels[1]._ThresholdBinary(new Gray(20), new Gray(255)); // Saturacion

            //Unir los filtros para obtener la imagen
            Image<Gray, byte> colordetimg = huefilter.And(valfilter).And(channels[1]);//aqui habia un Not()
            pictureBox2.Image = colordetimg.Bitmap;
            //colordetimg._Erode(1);
            Mat SE2 = CvInvoke.GetStructuringElement(Emgu.CV.CvEnum.ElementShape.Rectangle, new Size(3, 2), new Point(-1, -1));
            CvInvoke.MorphologyEx(colordetimg, colordetimg, MorphOp.Erode, SE2, new Point(-1, -1), 2, BorderType.Default, new MCvScalar(255));
            Mat SE3 = CvInvoke.GetStructuringElement(Emgu.CV.CvEnum.ElementShape.Rectangle, new Size(3, 3), new Point(-1, -1));
            CvInvoke.MorphologyEx(colordetimg, colordetimg, MorphOp.Dilate, SE3, new Point(-1, -1), 4, BorderType.Default, new MCvScalar(255));
           
            ////colordetimg._Erode(1);
            //colordetimg._Dilate(3);
            //colordetimg._Erode(1);

            //Colorear imagen
            //Image<Bgr, byte> ret = img;
            //var mat = img.Mat;
            //mat.SetTo(new MCvScalar(0, 0, 255), colordetimg);
            //mat.CopyTo(ret);
            //Image<Bgr, byte> imgout = ret.CopyBlank();//imagen sin fondo negro

            //ret._Or(imge);
            //Muestra imagen con los rojos destacados
            
            /*
             1. Edge detection (sobel)
             2. Dilation (10,1)
             3. FindContours
             4. Geometrical Constrints
             */
            //sobel
            Image<Gray, byte> sobel = colordetimg;
            //Image<Gray, byte> sobel = colordetimg.Sobel(0, 1, 1).AbsDiff(new Gray(0.0)).Convert<Gray, byte>();//.ThresholdBinary(new Gray(80), new Gray(255));
            //Image<Gray, byte> sobel2 = colordetimg.Sobel(1, 0, 1).AbsDiff(new Gray(0.0)).Convert<Gray, byte>();//.ThresholdBinary(new Gray(80), new Gray(255));
          
            //sobel.And(sobel2);
            //pictureBox3.Image = sobel.Bitmap;
            Mat SE = CvInvoke.GetStructuringElement(Emgu.CV.CvEnum.ElementShape.Rectangle, new Size(3, 3), new Point(-1, -1));
            CvInvoke.MorphologyEx(colordetimg,aux,  Emgu.CV.CvEnum.MorphOp.Close, SE, new Point(-1, -1),3, Emgu.CV.CvEnum.BorderType.Reflect, new MCvScalar(255));
            
            //CvInvoke.MorphologyEx(sobel,aux, MorphOp.Close, SE,new Point (-1,-1), 1, BorderType.Reflect, new MCvScalar(255));
            pictureBox3.Image = aux.Bitmap;
           
           

            _frame.CopyTo(finalFrame);
           
            DetectObject(aux, finalFrame, cropbox);


            pictureBox4.Image = finalFrame.Bitmap;
            //double fps = 15;
            //await Task.Delay(1000 / Convert.ToInt32(fps));

        }

       
    }
}