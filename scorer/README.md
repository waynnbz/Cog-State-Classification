# Marathon Scorer Template
Docker Template sample submission format using Python.

## Running Scorer Code
Build the container from within the root folder by
`docker build -t scorer-template .`

Launch the container with
`docker run -it scorer-template`

Verify that testing works out of the box. Within the container, run
Full Score: `./run-scorer.sh provisional provisional-truth.csv ./sample_submissions/full_score/solution/solution.csv .`
Partial Score: `./run-scorer.sh provisional provisional-truth.csv ./sample_submissions/partial_score/solution/solution.csv .`
Zero Score: `./run-scorer.sh provisional provisional-truth.csv ./sample_submissions/zero_score/solution/solution.csv .`

## What do I add to this repo?

Add your scorer code, required data files, and a Dockerfile.  Please build a 
docker image with your scorer code and verify that it can run as a docker 
container.

Your scorer must return the score that is to be posted to the challenge 
leaderboard.

Please add proper error handling to the scorer.

## Configuration Details

Please fill out detailed readme with requirements for scorer and instructions
on how to run the scorer. Here is the basic info that you need to provide.

1.  Planned start date of the match: 
1.  Docker Image Name: "topcoder/[YOUR_DOCKER_IMAGE NAME]:latest" 
1.  challengeId: 
1.  Command to run provisional scorer: For example: `./run-scorer.sh provisional truth-1.csv /workdir/solution/solution-1.csv .`
1.  Command to run final scorer: For example:`./run-scorer.sh final final-truth.csv /workdir/solution/solution.csv .`
1.  subCmdProvisional: NA
1.  subCmdFinal: command to run for final scoring IF they are submitting code.
1.  testTypeProvisional: [code|data]
1.  testTypeFinal: [code|data]
1.  timeoutProvisional: milliseconds to wait before provisional scoring times out: 
1.  timeoutFinal: milliseconds to wait before final scoring times out: 
1.  customRun: false
1.  customRunCmd: na
1.  customRunProvisional: "false"
1.  customRunFinal: "false"
1.  instanceType: c6g.large (default)
1.  volumeSize: 
1.  gpuEnableProvisional: false
1.  gpuEnableFinal: false


## FAQ

How much disk space is available to submitters to run their submissions?
- This is dictated by the `volumeSize` you specify in the above config

How long can a submission run until it times out?
- This is controlled by the timeout settings in the config

What folders are submissions allowed to write to?
- Submitters should make their submissions write to the `/workdir` folder.  All other folders are read only.

Are there any sample submissions or instructions submitters can reference?
- Yes. Please refer to the following repos for templates and detailed instructions on how to structure submissions:
1. [Code Submissions](https://github.com/topcoder-platform-templates/marathon-code-only)
1. [Data + Code Submissions](https://github.com/topcoder-platform-templates/marathon-data-and-code)
