/*
  Detects SIFT features in two images and finds matches between them.

  Copyright (C) 2006-2012  Rob Hess <rob@iqengines.com>

  @version 1.1.2-20100521
*/

#include "sift.h"
#include "imgfeatures.h"
#include "kdtree.h"
#include "utils.h"
#include "xform.h"

#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>

#include <stdio.h>
#include <unistd.h>

/* the maximum number of keypoint NN candidates to check during BBF search */
#define KDTREE_BBF_MAX_NN_CHKS 200

/* threshold on squared ratio of distances between NN and 2nd NN */
#define NN_SQ_DIST_RATIO_THR 0.49

#define OPTIONS ":o:m:i:s:c:r:n:b:dxh"

/*************************** Function Prototypes *****************************/

static void usage(char*);
static void arg_parse(int, char**);

/******************************** Globals ************************************/

char *pname;
char *img1_file_name;
char *img2_file_name;
char *out_file_name = NULL;
char *out_img_name = NULL;
int intvls = SIFT_INTVLS;
double sigma = SIFT_SIGMA;
double contr_thr = SIFT_CONTR_THR;
int curv_thr = SIFT_CURV_THR;
int img_dbl = SIFT_IMG_DBL;
int descr_width = SIFT_DESCR_WIDTH;
int descr_hist_bins = SIFT_DESCR_HIST_BINS;
int display = 1;


/********************************** Main *************************************/

int main(int argc, char **argv)
{
  IplImage *img1, *img2, *stacked;
  struct feature *feat1, *feat2, *feat;
  struct feature **nbrs;
  struct kd_node *kd_root;
  CvPoint pt1, pt2;
  double d0, d1;
  int n1, n2, k, i, m = 0;

  arg_parse(argc, argv);
  //if(argc != 3) fatal_error( "usage: %s <img1> <img2>", argv[0]);
  
  img1 = cvLoadImage(img1_file_name, 1);
  if(!img1) fatal_error("unable to load image from %s", img1_file_name);
  img2 = cvLoadImage(img2_file_name, 1 );
  if(!img2) fatal_error("unable to load image from %s", img2_file_name);
  stacked = stack_imgs(img1, img2);

  fprintf(stderr, "Finding features in %s...\n", img1_file_name);
  n1 = _sift_features(img1, &feat1, intvls, sigma, contr_thr, curv_thr,
		      img_dbl, descr_width, descr_hist_bins);
  fprintf(stderr, "Finding features in %s...\n", img2_file_name);
  n2 = _sift_features(img2, &feat2, intvls, sigma, contr_thr, curv_thr,
		      img_dbl, descr_width, descr_hist_bins);
  fprintf( stderr, "Building kd tree...\n" );
  kd_root = kdtree_build(feat2, n2);
  for(i = 0; i < n1; i++)
    {
      feat = feat1 + i;
      k = kdtree_bbf_knn( kd_root, feat, 2, &nbrs, KDTREE_BBF_MAX_NN_CHKS );
      if( k == 2 )
	{
	  d0 = descr_dist_sq( feat, nbrs[0] );
	  d1 = descr_dist_sq( feat, nbrs[1] );
	  if( d0 < d1 * NN_SQ_DIST_RATIO_THR )
	    {
	      pt1 = cvPoint(cvRound(feat->x), cvRound(feat->y));
	      pt2 = cvPoint(cvRound(nbrs[0]->x), cvRound(nbrs[0]->y));
	      pt2.y += img1->height;
	      cvLine(stacked, pt1, pt2, CV_RGB(255,0,255), 1, 8, 0);
	      m++;
	      feat1[i].fwd_match = nbrs[0];
	    }
	}
      free(nbrs);
    }

  fprintf(stderr, "Found %d total matches\n", m);

  if(out_file_name != NULL) export_features(out_file_name, feat, m);
  if(out_img_name != NULL) cvSaveImage(out_img_name, stacked, NULL);
  
  if(display) {
    draw_features(stacked, feat, m);
    display_big_img(stacked, "Matches");
    cvWaitKey(0);
  }

  /* 
     UNCOMMENT BELOW TO SEE HOW RANSAC FUNCTION WORKS
     
     Note that this line above:
     
     feat1[i].fwd_match = nbrs[0];
     
     is important for the RANSAC function to work.
  */
  /*
  {
    CvMat* H;
    IplImage* xformed;
    H = ransac_xform( feat1, n1, FEATURE_FWD_MATCH, lsq_homog, 4, 0.01,
		      homog_xfer_err, 3.0, NULL, NULL );
    if( H )
      {
	xformed = cvCreateImage(cvGetSize(img2), IPL_DEPTH_8U, 3);
	cvWarpPerspective( img1, xformed, H, 
			   CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS,
			   cvScalarAll(0));
	cvNamedWindow("Xformed", 1);
	cvShowImage("Xformed", xformed);
	cvWaitKey(0);
	cvReleaseImage(&xformed);
	cvReleaseMat(&H);
      }
  }
  */

  cvReleaseImage(&stacked);
  cvReleaseImage(&img1);
  cvReleaseImage(&img2);
  kdtree_release(kd_root);
  free(feat1);
  free(feat2);
  return 0;
}

/************************** Function Definitions *****************************/

// print usage for this program
static void usage(char *name)
{
  fprintf(stderr, "%s: match SIFT keypoints between two images\n\n", name);
  fprintf(stderr, "Usage: %s [options] <img1> <img2>\n", name);
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "  -h               Display this message and exit\n");
  fprintf(stderr, "  -o <out_file>    Output keypoints to text file\n");
  fprintf(stderr, "  -m <out_img>     Output keypoint image file (format" \
	  " determined by extension)\n");
  fprintf(stderr, "  -i <intervals>   Set number of sampled intervals per" \
	  " octave in scale space\n");
  fprintf(stderr, "                   pyramid (default %d)\n",
	  SIFT_INTVLS);
  fprintf(stderr, "  -s <sigma>       Set sigma for initial gaussian"	\
	  " smoothing at each octave\n");
  fprintf(stderr, "                   (default %06.4f)\n", SIFT_SIGMA);
  fprintf(stderr, "  -c <thresh>      Set threshold on keypoint contrast" \
	  " |D(x)| based on [0,1]\n");
  fprintf(stderr, "                   pixel values (default %06.4f)\n",
	  SIFT_CONTR_THR);
  fprintf(stderr, "  -r <thresh>      Set threshold on keypoint ratio of" \
	  " principle curvatures\n");
  fprintf(stderr, "                   (default %d)\n", SIFT_CURV_THR);
  fprintf(stderr, "  -n <width>       Set width of descriptor histogram" \
	  " array (default %d)\n", SIFT_DESCR_WIDTH);
  fprintf(stderr, "  -b <bins>        Set number of bins per histogram" \
	  " in descriptor array\n");
  fprintf(stderr, "                   (default %d)\n", SIFT_DESCR_HIST_BINS);
  fprintf(stderr, "  -d               Toggle image doubling (default %s)\n",
	  SIFT_IMG_DBL == 0 ? "off" : "on");
  fprintf(stderr, "  -x               Turn off keypoint display\n");
}



/*
  arg_parse() parses the command line arguments, setting appropriate globals.
  
  argc and argv should be passed directly from the command line
*/
static void arg_parse(int argc, char **argv)
{
  //extract program name from command line (remove path, if present)
  pname = basename(argv[0]);

  //parse commandline options
  while(1)
    {
      char *arg_check;
      int arg = getopt(argc, argv, OPTIONS);
      if(arg == -1)
	break;

      switch( arg )
	{
	  // catch unsupplied required arguments and exit
	case ':':
	  fatal_error( "-%c option requires an argument\n" \
	        "Try '%s -h' for help.", optopt, pname );
	  break;

	  // read out_file_name
	case 'o':
	  if( ! optarg )
	    fatal_error( "error parsing arguments at -%c\n"	\
			 "Try '%s -h' for help.", arg, pname );
	  out_file_name = optarg;
	  break;

	  // read out_img_name
	case 'm':
	  if( ! optarg )
	    fatal_error( "error parsing arguments at -%c\n"	\
			 "Try '%s -h' for help.", arg, pname );
	  out_img_name = optarg;
	  break;
	  
	  // read intervals
	case 'i':
	  // ensure argument provided
	  if( ! optarg )
	    fatal_error( "error parsing arguments at -%c\n"	\
			 "Try '%s -h' for help.", arg, pname );
	  
	  // parse argument and ensure it is an integer
	  intvls = strtol( optarg, &arg_check, 10 );
	  if( arg_check == optarg  ||  *arg_check != '\0' )
	    fatal_error( "-%c option requires an integer argument\n"	\
			 "Try '%s -h' for help.", arg, pname );
	  break;
	  
	  // read sigma
	case 's' :
	  // ensure argument provided
	  if( ! optarg )
	    fatal_error( "error parsing arguments at -%c\n"	\
			 "Try '%s -h' for help.", arg, pname );
	  
	  // parse argument and ensure it is a floating point number
	  sigma = strtod( optarg, &arg_check );
	  if( arg_check == optarg  ||  *arg_check != '\0' )
	    fatal_error( "-%c option requires a floating point argument\n" \
			 "Try '%s -h' for help.", arg, pname );
	  break;
	  
	  // read contrast_thresh
	case 'c' :
	  // ensure argument provided
	  if( ! optarg )
	    fatal_error( "error parsing arguments at -%c\n"	\
			 "Try '%s -h' for help.", arg, pname );
	  
	  // parse argument and ensure it is a floating point number
	  contr_thr = strtod( optarg, &arg_check );
	  if( arg_check == optarg  ||  *arg_check != '\0' )
	    fatal_error( "-%c option requires a floating point argument\n" \
			 "Try '%s -h' for help.", arg, pname );
	  break;
	  
	  // read curvature_thresh
	case 'r' :
	  // ensure argument provided
	  if( ! optarg )
	    fatal_error( "error parsing arguments at -%c\n"	\
			 "Try '%s -h' for help.", arg, pname );
	  
	  // parse argument and ensure it is a floating point number
	  curv_thr = strtol( optarg, &arg_check, 10 );
	  if( arg_check == optarg  ||  *arg_check != '\0' )
	    fatal_error( "-%c option requires an integer argument\n"	\
			 "Try '%s -h' for help.", arg, pname );
	  break;
	  
	  // read descr_width
	case 'n' :
	  // ensure argument provided
	  if( ! optarg )
	    fatal_error( "error parsing arguments at -%c\n"	\
			 "Try '%s -h' for help.", arg, pname );
	  
	  // parse argument and ensure it is a floating point number
	  descr_width = strtol( optarg, &arg_check, 10 );
	  if( arg_check == optarg  ||  *arg_check != '\0' )
	    fatal_error( "-%c option requires an integer argument\n"	\
			 "Try '%s -h' for help.", arg, pname );
	  break;
	  
	  // read descr_histo_bins
	case 'b' :
	  // ensure argument provided
	  if( ! optarg )
	    fatal_error( "error parsing arguments at -%c\n"	\
			 "Try '%s -h' for help.", arg, pname );
	  
	  // parse argument and ensure it is a floating point number
	  descr_hist_bins = strtol( optarg, &arg_check, 10 );
	  if( arg_check == optarg  ||  *arg_check != '\0' )
	    fatal_error( "-%c option requires an integer argument\n"	\
			 "Try '%s -h' for help.", arg, pname );
	  break;
	  
	  // read double_image
	case 'd' :
	  img_dbl = (img_dbl == 1)? 0 : 1;
	  break;

	  // read display
	case 'x' :
	  display = 0;
	  break;

	  // user asked for help
	case 'h':
	  usage( pname );
	  exit(0);
	  break;

	  // catch invalid arguments
	default:
	  fatal_error( "-%c: invalid option.\nTry '%s -h' for help.",
		       optopt, pname );
	}
    }

  // make sure an input file is specified
  if(argc - optind < 2) fatal_error("no input file specified.\nTry '%s -h' for help.", pname);

  // make sure there aren't too many arguments
  if(argc - optind > 2) fatal_error("too many arguments.\nTry '%s -h' for help.", pname );

  // copy image file name from command line argument
  img1_file_name = argv[optind];
  img2_file_name = argv[optind+1];
}
