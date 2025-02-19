[x] RTSP capturing
[x] Live feed capture and checking if its running
[x] Restart the feed capture if and when the stream fails
[x] Modelling / CV logic | This part might be owned by engineering in some please.
[x] Aggregation logic
[x] Alerts reporting
[x] Data saving logic
[-] Local data purging logics - If for any reason we are saving the data locally [for training, etc.]
[x] Ensuring persistent connection with S3/SQS/Mongo DB (whichever you are using)
[x] Script hardening - Try Except should be in place at appropriate places
-- Logging 
    [x] RTSP
    [x] Alert reporting
    [x] If connections to third party services fail or any other point where you think logging is needed.
[-] Results QA - For this DS might need to sit with the respective PM to confirm the results.
[-] Code documentation
[-] Project DS documentation on confluence