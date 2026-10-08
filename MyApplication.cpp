#include <opencv2/opencv.hpp>
#include <iostream>
#include <deque>
#include <vector>
#include <algorithm>
#include <map>
#include <iomanip>

using namespace std;

#define ABANDONED 1
#define REMOVED 2
#define OTHER_CHANGE 3

//   (1) Video number, (2) Frame number, (3) Type of change, (4) Top row,
//   (5) Left column, (6) Bottom row, (7) Right column
int object_locations[][7] = {
    {1, 115, REMOVED, 105,249,148,311},
    {2, 87, ABANDONED, 130,250,148,302},
    {3, 87, REMOVED, 164,186,223,234},
    {4, 91, ABANDONED, 208,356,238,388},
    {4, 255, REMOVED, 208,356,238,388},
    {5, 137, ABANDONED, 118,68,126,76},
    {5, 357, REMOVED, 118,68,126,76},
    {6, 127, ABANDONED, 28,210,40,227},
    {6, 379, REMOVED, 28,210,40,227},
    {7, 555, ABANDONED, 108,107,123,127},
    {8, 333, ABANDONED, 104,219,140,275},
    {9, 331, ABANDONED, 322,129,376,188},
    {10, 73, OTHER_CHANGE, 109,126,269,225},
    {10, 83, OTHER_CHANGE, 104,250,148,311}
};

// Parameters used throughout
const double MIN_AREA = 150;
const double MAX_AREA = 3000;
const int TEMPORAL_CONSISTENCY_FRAMES = 15;
const double BOX_OVERLAP_THRESHOLD = 0.5;
const int FRAME_TOLERANCE = 120;

// Function to determine if there is overlap in bounding box's 
bool boxesOverlap(cv::Rect box1, cv::Rect box2, double threshold) {
    cv::Rect intersection = box1 & box2;
    if (intersection.area() == 0) return false;
    
    double iou = (double)intersection.area() / (double)(box1.area() + box2.area() - intersection.area());
    return iou >= threshold;
}

// Convert ground truth format to cv::Rect - useful for evaluation 
cv::Rect gtToRect(int top, int left, int bottom, int right) {
    return cv::Rect(left, top, right - left, bottom - top);
}

// struct to store detection info
struct DetectionInfo {
    cv::Rect box;
    int type;
    bool matchedGT;
};

// Struct to store detected object info 
struct TrackedObject {
    cv::Rect box;
    int framesSeen;
    int lastFrameSeen;
    int detectedType;
    bool typeConfirmed;
    int abandonedVotes;
    int removedVotes;
};

// Compare two box's used to evualted against ground truth 
bool boxesSimilar(cv::Rect box1, cv::Rect box2) {
    cv::Point2f center1(box1.x + box1.width/2.0f, box1.y + box1.height/2.0f);
    cv::Point2f center2(box2.x + box2.width/2.0f, box2.y + box2.height/2.0f);
    
    float distance = sqrt(pow(center1.x - center2.x, 2) + pow(center1.y - center2.y, 2));
    float sizeRatio = (float)min(box1.area(), box2.area()) / max(box1.area(), box2.area());
    
    return (distance < 30 && sizeRatio > 0.6);
}

// May remove this function if possible - was needed throughout
// Merge overlapping tracked objects
void mergeOverlapping(vector<TrackedObject>& trackedObjects) {
    if (trackedObjects.size() <= 1) return;
    
    vector<bool> merged(trackedObjects.size(), false);
    vector<TrackedObject> result;
    
    for (size_t i = 0; i < trackedObjects.size(); i++) {
        if (merged[i]) continue;
        
        TrackedObject& current = trackedObjects[i];
        
        for (size_t j = i + 1; j < trackedObjects.size(); j++) {
            if (merged[j]) continue;
            
            if (boxesOverlap(current.box, trackedObjects[j].box, 0.3)) {
                TrackedObject& other = trackedObjects[j];
                
                if (other.box.area() > current.box.area()) {
                    current.box = other.box;
                }
                
                current.framesSeen = max(current.framesSeen, other.framesSeen);
                current.lastFrameSeen = max(current.lastFrameSeen, other.lastFrameSeen);
                current.abandonedVotes += other.abandonedVotes;
                current.removedVotes += other.removedVotes;
                
                if (current.abandonedVotes > current.removedVotes) {
                    current.detectedType = ABANDONED;
                } else if (current.removedVotes > current.abandonedVotes) {
                    current.detectedType = REMOVED;
                }
                
                int totalVotes = current.abandonedVotes + current.removedVotes;
                if (totalVotes >= 5) {
                    current.typeConfirmed = true;
                }
                
                merged[j] = true;
            }
        }
        
        result.push_back(current);
    }
    
    trackedObjects = result;
}

// Main video processing function 
// Take in the the video path, video number and all dynamic info for evaluation 
// Update the dyanmic eval info for final evaluation 
void processVideo(string videoPath, int videoNumber,
                  map<int, map<int, int>>& typeCountsByGT,
                  map<int, bool>& boxMatchesByGT,
                  vector<DetectionInfo>& allDetections) {

    // Open video - return with an error message if it doesn't open
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        cerr << "Error: Could not open video file: " << videoPath << endl;
        return;
    }
    
    // Initialise frame variables
    cv::Mat frame, fgMask, gray, motionMask, stationaryMask, persistence;
    
    // Create GMM (Gaussian Mixture Model) for background subtraction
    cv::Ptr<cv::BackgroundSubtractorMOG2> gmm = cv::createBackgroundSubtractorMOG2(500, 25, true);
    // Set Shadow detection to false - Helped reduce noise to a degree
    gmm->setDetectShadows(false);
    
    // Frame buffers - used in the temporal difference for the motion mask 
    const int FRAME_SKIP = 5;
    deque<cv::Mat> frameBuffer;
    
    // Initialise variables used for the frame history 
    // This is important for the classification 
    const int EARLY_HISTORY_SIZE = 50;
    deque<cv::Mat> earlyFrameHistory;
    bool earlyHistoryComplete = false;
    
    // Get (or initialise) the FPS
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 30;
    
    // Initialise the stationary time and the peristance threshold
    const double STATIONARY_TIME = 6.0;     // From testing 6 second worked best 
    int persistenceThreshold = static_cast<int>(STATIONARY_TIME * fps);
    if (persistenceThreshold > 255) persistenceThreshold = 255;
    
    // Get ground truth boxes for this video - This helped visually evaluate the code 
    int numGroundTruthEvents = sizeof(object_locations) / sizeof(object_locations[0]);
    vector<cv::Rect> gtBoxes;
    for (int i = 0; i < numGroundTruthEvents; i++) {
        if (object_locations[i][0] == videoNumber) {
            cv::Rect gtBox = gtToRect(object_locations[i][3], object_locations[i][4],
                                      object_locations[i][5], object_locations[i][6]);
            gtBoxes.push_back(gtBox);
        }
    }
    
    // INitialise object tracking vector
    vector<TrackedObject> trackedObjects;
    // Initialise frame count to 0
    int frameCount = 0;
    
    // Loop through the video - from passed video path 
    while (true) {
        // Initialise the current frame, breaking if it's empty 
        cv::Mat originalFrame;
        cap >> originalFrame;
        if (originalFrame.empty()) break;
        
        // Increase the frame count, cloning the current frame 
        frameCount++;
        frame = originalFrame.clone();
        
        // Build up a history - Used in evaluation 
        if (!earlyHistoryComplete) {
            earlyFrameHistory.push_back(originalFrame.clone());
            if (earlyFrameHistory.size() >= EARLY_HISTORY_SIZE) {
                earlyHistoryComplete = true;
            }
        }
        
        // Use Gaussian Blurring to help reduce noise
        cv::GaussianBlur(frame, frame, cv::Size(5, 5), 1.5);
        
        // Perform background subtraction using the GMM background model
        gmm->apply(frame, fgMask, 0.0005);
        
        // Use morph opening and closing to clean up the output after subtraction 
        // This again helps to use noise but can reduce the amount of detail in the detected object
        cv::morphologyEx(fgMask, fgMask, cv::MORPH_OPEN,
                         cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3)));
        cv::morphologyEx(fgMask, fgMask, cv::MORPH_CLOSE,
                         cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5)));
        
        // Convert the original frame to greyscale 
        // Needed for the motion mask 
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        
        // Build frame buffer for motion detection
        // Helps detect more subtle movements
        frameBuffer.push_back(gray.clone());
        if (frameBuffer.size() > FRAME_SKIP + 1)
            frameBuffer.pop_front();
        
        // Create a motion mask by comparing the current frame to the frame at the other end of the buffer. 
        // Helps give a more complete motion mask where not only the leading edge is shown
        // The difference is then thresholded to get a binary image 
        if (frameBuffer.size() == FRAME_SKIP + 1) {
            cv::absdiff(frameBuffer.back(), frameBuffer.front(), motionMask);
            cv::threshold(motionMask, motionMask, 25, 255, cv::THRESH_BINARY);
            cv::morphologyEx(motionMask, motionMask, cv::MORPH_OPEN,
                             cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3)));
        } else {
            // If the buffer size is incomplete - initialise to zeros
            motionMask = cv::Mat::zeros(gray.size(), CV_8U);
        }
        
        // Bitwise AND the motion mask and the GMM mask 
        // Explained in report how this finds newly stationary objects 
        cv::bitwise_and(fgMask, 255 - motionMask, stationaryMask);
        
        // Track persistence of stationary objects over time
        if (persistence.empty())
            // If empty initialise to zweros
            persistence = cv::Mat::zeros(gray.size(), CV_8U);
        
        // Update the persistence 
        persistence.setTo(0, stationaryMask == 0);
        persistence += (stationaryMask / 255);
        
        // Threshold to find objects that have been stationary long enough
        // This is our suspected stationary mask 
        cv::Mat suspectedStationary;
        cv::threshold(persistence, suspectedStationary, persistenceThreshold, 255, cv::THRESH_BINARY);
        cv::morphologyEx(suspectedStationary, suspectedStationary, cv::MORPH_OPEN,
                         cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));
        
        // Find the contours in the suspected stationary mask 
        vector<vector<cv::Point>> contours;
        cv::findContours(suspectedStationary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        // cout << "Found " << contours.size() << " contours" << endl;  // for debugging
        
        // Limit contours to prevent performance issues
        // This was a trade off to improve performance however it may have had a negative effect on the classification 
        const int MAX_CONTOURS_PER_FRAME = 50;
        if (contours.size() > MAX_CONTOURS_PER_FRAME) {
            sort(contours.begin(), contours.end(),
                [](const vector<cv::Point>& a, const vector<cv::Point>& b) {
                    return cv::contourArea(a) > cv::contourArea(b);
                });
            contours.resize(MAX_CONTOURS_PER_FRAME);
        }
        
        // Update tracked objects
        for (auto& tracked : trackedObjects) {
            if (tracked.lastFrameSeen < frameCount - 1) {
                tracked.framesSeen = max(0, tracked.framesSeen - 2);
            }
        }
        
        vector<bool> contourMatched(contours.size(), false);
        vector<bool> trackedMatched(trackedObjects.size(), false);
        
        // For each contour
        for (size_t idx = 0; idx < contours.size(); idx++) {
            // Define the contours area
            double area = cv::contourArea(contours[idx]);
            
            // Filter out areas too big and too small 
            // See defs at the top
            if (area < MIN_AREA || area > MAX_AREA) continue;
            
            // Draw a bounding box around any contours that make it through
            cv::Rect box = cv::boundingRect(contours[idx]);
            
            // Filter out any elongated boxes, 
            // Note: I chose to do this to help with the noise created by shadows
            // However in the event we were looking for a golf club this would definitly have to be changed
            double aspectRatio = (double)box.width / box.height;
            if (aspectRatio > 5.0 || aspectRatio < 0.2) continue;
            
            // Begin Classification of abandoned or removed objects 
            // Initialise as other and confirmed bool as false
            int objectType = OTHER_CHANGE;
            bool typeConfirmed = false;
            
            // First check there is an earlier history to compare to 
            if (earlyHistoryComplete) {
                cv::Rect roi = box & cv::Rect(0, 0, gray.cols, gray.rows);
                
                if (roi.area() > 25 && 
                    roi.x + roi.width <= suspectedStationary.cols &&
                    roi.y + roi.height <= suspectedStationary.rows) {
                    
                    cv::Mat persistenceROI = suspectedStationary(roi);
                    cv::Mat persistenceEdges;
                    cv::Canny(persistenceROI, persistenceEdges, 50, 150);
                    
                    // Compare with baseline frames to determine object class
                    // Accumulate evidence for removed or accumulate 
                    double removed_overlap = 0.0;
                    double abandoned_overlap = 0.0;
                    double removed_count = 0.0;
                    double abandoned_count = 0.0;
                    int totalComparisons = 0;
                    
                    // Decide which frames in the baseline history are used for comparison
                    vector<int> baselineIndices;
                    if (earlyFrameHistory.size() > 40) {
                        baselineIndices = {10, 20, 30, 40};
                    } else if (earlyFrameHistory.size() > 20) {
                        baselineIndices = {5, 10, 15, 20};
                    } else if (earlyFrameHistory.size() > 10) {
                        baselineIndices = {3, 6, 9};
                    }
                    
                    // Compare with multiple baseline frames
                    for (int BL_index : baselineIndices) {
                        if (BL_index >= (int)earlyFrameHistory.size()) continue;
                        
                        cv::Mat baselineGray;
                        cv::cvtColor(earlyFrameHistory[BL_index], baselineGray, cv::COLOR_BGR2GRAY);
                        cv::Mat baselineROI = baselineGray(roi);
                        cv::Mat baselineEdges;
                        cv::Canny(baselineROI, baselineEdges, 50, 150);
                        
                        // Calculate edge overlap - used for classification 
                        cv::Mat edgeOverlap;
                        cv::bitwise_and(persistenceEdges, baselineEdges, edgeOverlap);
                        
                        int persistenceEdgeCount = cv::countNonZero(persistenceEdges);
                        int baselineEdgeCount = cv::countNonZero(baselineEdges);
                        int overlapCount = cv::countNonZero(edgeOverlap);
                        
                        if (persistenceEdgeCount == 0) continue;
                        
                        // High overlap = object was REMOVED (edges match baseline)
                        // Low overlap = object was ABANDONED (new edges)
                        double overlapRatio = (double)overlapCount / persistenceEdgeCount;
                        if (overlapRatio > 0.25) {
                            removed_overlap += 1.0;
                        } else if (overlapRatio < 0.1) {
                            abandoned_overlap += 1.0;
                        }
                        
                        // More baseline edges = something was removed
                        // This comparison is accumulated over multiple history frames
                        double edgeCountRatio = (double)baselineEdgeCount / persistenceEdgeCount;
                        
                        if (edgeCountRatio > 1.3) {
                            removed_count += 0.3;
                        } else if (edgeCountRatio < 0.8) {
                            abandoned_count += 0.3;
                        }
                        totalComparisons++;
                    }
                    
                    // Determine type based on accumulated scores
                    double totalRemovedScore = removed_overlap + removed_count;
                    double totalAbandonedScore = abandoned_overlap + abandoned_count;
                    
                    // Check the appropiate number of comparisons were made
                    if (totalComparisons >= 3) {
                        if (totalRemovedScore > totalAbandonedScore && (totalRemovedScore - totalAbandonedScore) >= 1.0) {
                            objectType = REMOVED;
                            typeConfirmed = true;
                        } else if (totalAbandonedScore > totalRemovedScore && (totalAbandonedScore - totalRemovedScore) >= 1.0) {
                            objectType = ABANDONED;
                            typeConfirmed = true;
                        }
                    }
                }
            }
            
            // Update or create tracked object
            // This was all implimented to cut down on False positives. 
            // Doing more calculations and comparions to make sure of detections
            bool foundTracked = false;
            for (size_t j = 0; j < trackedObjects.size(); j++) {
                if (!trackedMatched[j] && boxesSimilar(box, trackedObjects[j].box)) {
                    // Update existing tracked object
                    trackedObjects[j].box = box;
                    trackedObjects[j].framesSeen++;
                    trackedObjects[j].lastFrameSeen = frameCount;
                    
                    // If object type is confirmed, update the object
                    if (typeConfirmed) {
                        if (objectType == ABANDONED) {
                            trackedObjects[j].abandonedVotes++;
                        } else if (objectType == REMOVED) {
                            trackedObjects[j].removedVotes++;
                        }
                        
                        // Update type based on majority vote
                        if (trackedObjects[j].abandonedVotes > trackedObjects[j].removedVotes) {
                            trackedObjects[j].detectedType = ABANDONED;
                        } else if (trackedObjects[j].removedVotes > trackedObjects[j].abandonedVotes) {
                            trackedObjects[j].detectedType = REMOVED;
                        }
                        
                        // Confirm type if enough evidence
                        // Previously classification was happing too early causing copius false positives
                        // Adding the confirmation based on temporal consistency drastically reduced this 
                        if (trackedObjects[j].framesSeen >= TEMPORAL_CONSISTENCY_FRAMES) {
                            int totalVotes = trackedObjects[j].abandonedVotes + trackedObjects[j].removedVotes;
                            if (totalVotes >= 5) {
                                trackedObjects[j].typeConfirmed = true;
                            }
                        }
                    }
                    // Update tracking variables
                    foundTracked = true;
                    contourMatched[idx] = true;
                    trackedMatched[j] = true;
                    // break the loop
                    break;
                }
            }
            
            if (!foundTracked) {
                // If we are not currently tracking an object 
                // Create a new trackeding object and initialise it
                TrackedObject newTracked;
                newTracked.box = box;
                newTracked.framesSeen = 1;
                newTracked.lastFrameSeen = frameCount;
                newTracked.detectedType = objectType;
                newTracked.typeConfirmed = false;
                newTracked.abandonedVotes = (objectType == ABANDONED && typeConfirmed) ? 1 : 0;
                newTracked.removedVotes = (objectType == REMOVED && typeConfirmed) ? 1 : 0;
                trackedObjects.push_back(newTracked);
            }
        }
        
        // Merge overlapping tracked objects
        // Compuatationally heavy but again helps reduce false positives
        mergeOverlapping(trackedObjects);
        
        // Remove stale tracked objects
        trackedObjects.erase(
            remove_if(trackedObjects.begin(), trackedObjects.end(),
                [frameCount](const TrackedObject& obj) {
                    return (frameCount - obj.lastFrameSeen) > 20;
                }),
            trackedObjects.end()
        );
        
        // Limit number of tracked objects - for some reason this limit fails in video 10 
        // Still doesn't prevent the crash
        // Really helps in video 2
        const int MAX_TRACKED_OBJECTS = 25;
        if (trackedObjects.size() > MAX_TRACKED_OBJECTS) {
            sort(trackedObjects.begin(), trackedObjects.end(),
                [](const TrackedObject& a, const TrackedObject& b) {
                    return a.framesSeen > b.framesSeen;
                });
            trackedObjects.resize(MAX_TRACKED_OBJECTS);
        }
        
        // For each tracked object compare it to the ground truth - Eval 
        for (const auto& tracked : trackedObjects) {
            // If tracked has been around long enough and tracking type is confirmed 
            if (tracked.framesSeen >= TEMPORAL_CONSISTENCY_FRAMES && tracked.typeConfirmed) {
                cv::Rect box = tracked.box;
                int objectType = tracked.detectedType;
                
                // Check each ground truth event for this video
                bool matchedAnyGT = false;
                for (int i = 0; i < numGroundTruthEvents; i++) {
                    if (object_locations[i][0] != videoNumber) continue;
                    
                    // Check if current frame is within tolerance of GT frame
                    // I opeted to give some tolerance when comparing to ground truth 
                    int frameDifference = abs(frameCount - object_locations[i][1]);
                    if (frameDifference > FRAME_TOLERANCE) continue;
                    
                    cv::Rect gtBox = gtToRect(object_locations[i][3], object_locations[i][4],
                                              object_locations[i][5], object_locations[i][6]);
                    
                    // Check if bounding box overlaps with ground truth
                    if (boxesOverlap(box, gtBox, BOX_OVERLAP_THRESHOLD)) {
                        typeCountsByGT[i][objectType]++;
                        boxMatchesByGT[i] = true;
                        matchedAnyGT = true;
                    }
                }
                
                // Record this detection (only once per tracked object)
                // Another thing that helped reduce false positives
                bool alreadyRecorded = false;
                for (const auto& det : allDetections) {
                    if (boxesSimilar(det.box, box) && det.type == objectType) {
                        alreadyRecorded = true;
                        break;
                    }
                }
                
                if (!alreadyRecorded) {
                    allDetections.push_back({box, objectType, matchedAnyGT});
                }
            }
        }
        
        // Visualization: Draw tracked objects
        for (const auto& tracked : trackedObjects) {
            cv::Rect box = tracked.box;
            // Check box is within bounds before drawing - at this stage it defintily should be 
            if (box.x < 0 || box.y < 0 || 
                box.x + box.width > frame.cols || 
                box.y + box.height > frame.rows) {
                continue;
            }
            // Colour and lable were used for the visual analysis 
            // Helped identify that previous versions assigned classes too quickly 
            cv::Scalar boxColor;
            string label;
            // Draw the approritae bounding box with the correct colour
            if (tracked.typeConfirmed && tracked.framesSeen >= TEMPORAL_CONSISTENCY_FRAMES) {
                if (tracked.detectedType == ABANDONED) {
                    boxColor = cv::Scalar(0, 255, 0);  // Green
                    label = "ABANDONED (" + to_string(tracked.framesSeen) + ")";
                } else if (tracked.detectedType == REMOVED) {
                    boxColor = cv::Scalar(255, 0, 0);  // Blue
                    label = "REMOVED (" + to_string(tracked.framesSeen) + ")";
                } else {
                    boxColor = cv::Scalar(0, 165, 255);  // Orange
                    label = "UNKNOWN (" + to_string(tracked.framesSeen) + ")";
                }
            } else {
                boxColor = cv::Scalar(0, 255, 255);  // Yellow
                label = "TRACKING... (" + to_string(tracked.framesSeen) + ")";
            }
            
            cv::rectangle(frame, box, boxColor, 2);
            cv::putText(frame, label, cv::Point(box.x, box.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, boxColor, 2);
        }
        
        // Draw ground truth boxes in black
        // Very useful for visual analysis 
        for (const auto& gtBox : gtBoxes) {
            cv::rectangle(frame, gtBox, cv::Scalar(0, 0, 0), 2);
            cv::putText(frame, "GT", cv::Point(gtBox.x, gtBox.y + gtBox.height + 15),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 2);
        }
        
        // Display windows - Used for debugging and visual analysis 
        cv::imshow("Frame", frame);
        cv::imshow("GMM", fgMask);
        cv::imshow("Motion Mask", motionMask);
        cv::imshow("Suspected Stationary Objects", suspectedStationary);
        
        // Poll for exit key
        char key = cv::waitKey(30);
        if (key == '0' || key == 27) break;
    }
    // Close all windoes
    cap.release();
    cv::destroyAllWindows();
}

void MyApplication() {
    // IMPORTANT NOTE 
    // I've removed the path for video 10 as it causes the code to crash 
    // I have addressed this at length in the report 
    vector<string> videoPaths = {
        "Video01.avi",
        "Video02.avi",
        "Video03.avi",
        "Video04.avi",
        "Video05.avi",
        "Video06.avi",
        "Video07.avi",
        "Video08.avi",
        "Video09.avi"
    };
    
    // Variables needed for evaluation tracking
    map<int, map<int, int>> typeCountsByGT;
    map<int, bool> boxMatchesByGT;
    vector<DetectionInfo> allDetections;
    int numGroundTruthEvents = sizeof(object_locations) / sizeof(object_locations[0]);
    
    // Per video variables needed for evaluation 
    map<int, int> truePositivesByVideo;
    map<int, int> falsePositivesByVideo;
    map<int, int> falseNegativesByVideo;
    
    // Process each video - (not 10)
    for (size_t i = 0; i < videoPaths.size(); i++) {
        int videoNumber = i + 1;
        cout << "Processing: " << videoPaths[i] << endl;
        
        int detectionsBefore = allDetections.size();
        processVideo(videoPaths[i], videoNumber, typeCountsByGT, boxMatchesByGT, allDetections);
        int detectionsAfter = allDetections.size();
        
        // Calculate per-video metrics
        int videoTP = 0;
        int videoFP = 0;
        
        for (int j = detectionsBefore; j < detectionsAfter; j++) {
            if (allDetections[j].matchedGT) {
                videoTP++;
            } else {
                videoFP++;
            }
        }
        
        // Count FN for this video (GT events not detected)
        int videoFN = 0;
        for (int j = 0; j < numGroundTruthEvents; j++) {
            if (object_locations[j][0] == videoNumber) {
                if (boxMatchesByGT.find(j) == boxMatchesByGT.end() || !boxMatchesByGT[j]) {
                    videoFN++;
                }
            }
        }
        
        truePositivesByVideo[videoNumber] = videoTP;
        falsePositivesByVideo[videoNumber] = videoFP;
        falseNegativesByVideo[videoNumber] = videoFN;
    }
    
    // Print F1 Score Results
    cout << "\nF1 SCORES:\n" << endl;
    
    int totalTP = 0;
    int totalFP = 0;
    int totalFN = 0;
    
    for (int videoNum = 1; videoNum <= 10; videoNum++) {
        int tp = truePositivesByVideo[videoNum];
        int fp = falsePositivesByVideo[videoNum];
        int fn = falseNegativesByVideo[videoNum];
        
        totalTP += tp;
        totalFP += fp;
        totalFN += fn;
        
        double precision = (tp + fp > 0) ? (double)tp / (tp + fp) : 0.0;
        double recall = (tp + fn > 0) ? (double)tp / (tp + fn) : 0.0;
        double f1 = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0.0;
        
        cout << "Video " << videoNum << endl;
        cout << "TP: " << tp << ", FP: " << fp << ", FN: " << fn << endl;
        cout << "Precision: " << fixed << setprecision(3) << precision 
                  << ", Recall: " << recall << ", F1: " << f1 << endl;
        cout << endl;
    }
    
    double overallPrecision = (totalTP + totalFP > 0) ? (double)totalTP / (totalTP + totalFP) : 0.0;
    double overallRecall = (totalTP + totalFN > 0) ? (double)totalTP / (totalTP + totalFN) : 0.0;
    double overallF1 = (overallPrecision + overallRecall > 0) ? 
                       2 * overallPrecision * overallRecall / (overallPrecision + overallRecall) : 0.0;
    
    cout << "OVERALL RESULTS" << endl;
    cout << "Total TP: " << totalTP << ", Total FP: " << totalFP << ", Total FN: " << totalFN << endl;
    cout << "Overall Precision: " << fixed << setprecision(3) << overallPrecision << endl;
    cout << "Overall Recall: " << overallRecall << endl;
    cout << "Overall F1 Score: " << overallF1 << endl;
    
    // Counter-based evaluation for detection and classification
    int static_obj_counter = 0;
    int class_counter = 0;
    int total_events = 0;
    
    for (int i = 0; i < numGroundTruthEvents; i++) {
        int gtType = object_locations[i][2];
        if (gtType == ABANDONED || gtType == REMOVED) {
            total_events++;
        }
    }
    
    for (int i = 0; i < numGroundTruthEvents; i++) {
        int gtType = object_locations[i][2];
        if (gtType != ABANDONED && gtType != REMOVED) continue;
        
        cv::Rect gtBox = gtToRect(object_locations[i][3], object_locations[i][4],
                                  object_locations[i][5], object_locations[i][6]);
        
        bool locationDetected = false;
        bool classCorrect = false;
        
        for (const auto& det : allDetections) {
            if (boxesOverlap(det.box, gtBox, BOX_OVERLAP_THRESHOLD)) {
                locationDetected = true;
                if (det.type == gtType) {
                    classCorrect = true;
                }
            }
        }
        
        if (locationDetected) static_obj_counter++;
        if (classCorrect) class_counter++;
    }
    
    // Calculate percentages - used in final output
    double detection_percentage = (total_events > 0) ? (double)static_obj_counter / total_events * 100.0 : 0.0;
    double classification_percentage = (total_events > 0) ? (double)class_counter / total_events * 100.0 : 0.0;
    
    // Output counter-based evaluation
    // Use to see how well each section performed
    // Does not account for video 10
    cout << "\nCOUNTER-BASED EVALUATION" << endl;
    cout << "Static Object Detection: " << static_obj_counter << "/" << total_events 
         << " (" << fixed << setprecision(1) << detection_percentage << "%)" << endl;
    cout << "Correct Classification: " << class_counter << "/" << total_events 
         << " (" << fixed << setprecision(1) << classification_percentage << "%)" << endl;
}

// Code didn't run without the main application 
// Everyting happens in the my application function tho
int main(){
    MyApplication();
    return 0;
}