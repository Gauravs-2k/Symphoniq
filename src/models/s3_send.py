#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S3 Upload Utility - Upload all files from a directory to an S3 bucket
with support for generating signed URLs for file download
"""

import os
import sys
import argparse
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from pathlib import Path
from tqdm import tqdm
import mimetypes
import concurrent.futures
import datetime
import time

# Hardcoded S3 configuration - CHANGE THESE VALUES
S3_CONFIG = {
    'bucket': 'iit-symphoniq',
    'aws_access_key_id': 'AKIAQUFLP4IVKRLIXANY',
    'aws_secret_access_key': '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',
    'region_name': 'ap-south-1',
    'default_prefix': 'uploads/flute/midi'  # Default prefix for uploads
}

def verify_s3_credentials():
    """Verify S3 credentials are valid by testing a simple operation"""
    s3_client = boto3.client(
        's3', 
        aws_access_key_id=S3_CONFIG['aws_access_key_id'],
        aws_secret_access_key=S3_CONFIG['aws_secret_access_key'],
        region_name=S3_CONFIG['region_name']
    )
    
    try:
        # Try a simple operation - list the first object in the bucket
        response = s3_client.list_objects_v2(
            Bucket=S3_CONFIG['bucket'],
            MaxKeys=1
        )
        
        # If we get here without an exception, credentials are valid
        print("✅ AWS credentials verified successfully")
        return True
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_message = e.response.get('Error', {}).get('Message', str(e))
        
        print(f"❌ AWS credentials verification failed: {error_code} - {error_message}")
        
        if (error_code == 'SignatureDoesNotMatch'):
            print("\n🔍 TROUBLESHOOTING SIGNATURE MISMATCH:")
            print("1. Check that your AWS secret key is correct (no extra spaces or typos)")
            print("2. Verify your system clock is accurate (time synchronization issues can cause this error)")
            print("3. Make sure you're using the correct region for your bucket")
            
            # Check system time
            aws_time = None
            try:
                # Try to get AWS time using STS service
                sts = boto3.client('sts', 
                    region_name=S3_CONFIG['region_name'],
                    aws_access_key_id=S3_CONFIG['aws_access_key_id'],
                    aws_secret_access_key=S3_CONFIG['aws_secret_access_key']
                )
                response = sts.get_caller_identity()
                aws_time = datetime.datetime.strptime(
                    response['ResponseMetadata']['HTTPHeaders']['date'],
                    '%a, %d %b %Y %H:%M:%S %Z'
                )
            except Exception:
                pass
                
            local_time = datetime.datetime.now(datetime.timezone.utc)
            if aws_time:
                time_diff = (local_time - aws_time).total_seconds()
                print(f"   - Your system time: {local_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
                print(f"   - AWS server time:  {aws_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
                print(f"   - Time difference:  {abs(time_diff):.1f} seconds")
                
                if abs(time_diff) > 300:  # 5 minutes
                    print("   ⚠️ Your system clock is significantly different from AWS time!")
                    print("      Please synchronize your system clock and try again.")
            
        elif error_code == 'InvalidAccessKeyId':
            print("\n🔍 The access key ID is invalid. Please check for typos.")
        
        elif error_code == 'AccessDenied':
            print("\n🔍 Your credentials are valid but you don't have permission to access this bucket.")
            print("   Check IAM permissions for this user and bucket.")
            
        return False
        
    except NoCredentialsError:
        print("❌ No AWS credentials found")
        return False
        
    except Exception as e:
        print(f"❌ Error verifying credentials: {str(e)}")
        return False

def initialize_s3_client():
    """Initialize and return an S3 client using hardcoded credentials"""
    try:
        return boto3.client(
            's3', 
            aws_access_key_id=S3_CONFIG['aws_access_key_id'],
            aws_secret_access_key=S3_CONFIG['aws_secret_access_key'],
            region_name=S3_CONFIG['region_name']
        )
    except Exception as e:
        print(f"Error initializing S3 client: {e}")
        sys.exit(1)

def get_file_mime_type(file_path):
    """Get the MIME type of a file"""
    mime_type, _ = mimetypes.guess_type(file_path)
    if (mime_type is None):
        # Default to binary if MIME type cannot be determined
        mime_type = 'application/octet-stream'
    return mime_type

def upload_file(file_path, bucket, s3_key, s3_client):
    """Upload a single file to S3"""
    try:
        mime_type = get_file_mime_type(file_path)
        s3_client.upload_file(
            str(file_path), 
            bucket, 
            s3_key,
            ExtraArgs={'ContentType': mime_type}
        )
        return True, s3_key
    except ClientError as e:
        return False, f"Error uploading {file_path}: {e}"

def list_all_files(directory_path, include_subdirs=True):
    """List all files in a directory and optionally its subdirectories"""
    directory_path = Path(directory_path)
    
    if not directory_path.exists():
        print(f"Error: Directory {directory_path} does not exist.")
        return []
    
    all_files = []
    
    if include_subdirs:
        # Walk through all subdirectories
        for root, _, files in os.walk(directory_path):
            for file in files:
                all_files.append(Path(root) / file)
    else:
        # Only get files in the top-level directory
        all_files = [f for f in directory_path.iterdir() if f.is_file()]
    
    return all_files

def upload_directory_to_s3(directory_path, s3_prefix="", include_subdirs=True, max_workers=10):
    """Upload all files in a directory to S3"""
    # Initialize S3 client
    s3_client = initialize_s3_client()
    bucket = S3_CONFIG['bucket']
    
    # Normalize the directory path
    directory_path = Path(directory_path).resolve()
    
    # Get all files
    files = list_all_files(directory_path, include_subdirs)
    
    if not files:
        print(f"No files found in {directory_path}")
        return []
    
    print(f"Found {len(files)} files to upload")
    
    # Prepare S3 keys and file paths
    uploads = []
    for file_path in files:
        # Create relative path from the directory
        relative_path = file_path.relative_to(directory_path)
        # Create S3 key by joining s3_prefix with relative path
        s3_key = str(Path(s3_prefix) / relative_path).replace('\\', '/')
        uploads.append((str(file_path), s3_key))
    
    # Upload files with progress bar
    successful = []
    failed = []
    
    # Use ThreadPoolExecutor for parallel uploads
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a dictionary mapping futures to their file info
        future_to_file = {
            executor.submit(upload_file, file_path, bucket, s3_key, s3_client): (file_path, s3_key)
            for file_path, s3_key in uploads
        }
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_file), 
                          total=len(future_to_file), desc="Uploading files"):
            file_path, s3_key = future_to_file[future]
            try:
                success, result = future.result()
                if success:
                    successful.append(result)
                else:
                    failed.append(result)
                    print(f"Failed: {result}")
            except Exception as e:
                failed.append(f"Exception uploading {file_path}: {e}")
                print(f"Exception: {e}")
    
    # Print summary
    print(f"\nUpload Summary:")
    print(f"  - Total files: {len(uploads)}")
    print(f"  - Successfully uploaded: {len(successful)}")
    print(f"  - Failed: {len(failed)}")
    
    # Generate manifest file
    if successful:
        manifest_path = Path(directory_path) / "s3_manifest.txt"
        with open(manifest_path, 'w') as f:
            for s3_key in successful:
                s3_uri = f"s3://{bucket}/{s3_key}"
                f.write(f"{s3_uri}\n")
        print(f"\nGenerated manifest file: {manifest_path}")
    
    return successful

def download_file(s3_key, output_path=None, bucket=None):
    """
    Download a file from S3 directly
    
    Parameters:
    s3_key (str): S3 object key (path to file in S3)
    output_path (str): Local path to save the file (default: use filename from S3 key)
    bucket (str): Optional bucket name, uses default if not provided
    
    Returns:
    bool: True if download was successful, False otherwise
    """
    if bucket is None:
        bucket = S3_CONFIG['bucket']
        
    s3_client = initialize_s3_client()
    
    # If no output path specified, use the filename from the S3 key
    if output_path is None:
        output_path = os.path.basename(s3_key)
    
    try:
        print(f"Downloading s3://{bucket}/{s3_key} to {output_path}...")
        s3_client.download_file(bucket, s3_key, output_path)
        print(f"✅ Download complete: {output_path}")
        return True
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_message = e.response.get('Error', {}).get('Message', str(e))
        print(f"❌ Error downloading file: {error_code} - {error_message}")
        return False
    except Exception as e:
        print(f"❌ Error downloading file: {str(e)}")
        return False

def generate_public_url(s3_key, bucket=None):
    """
    Generate a public URL for accessing a file from S3
    Note: This only works if the bucket/object is configured with public read access
    
    Parameters:
    s3_key (str): S3 object key (path to file in S3)
    bucket (str): Optional bucket name, uses default if not provided
    
    Returns:
    str: Public URL for the file
    """
    if bucket is None:
        bucket = S3_CONFIG['bucket']
    
    region = S3_CONFIG['region_name']
    
    # Format: https://{bucket}.s3.{region}.amazonaws.com/{object_key}
    public_url = f"https://{bucket}.s3.{region}.amazonaws.com/{s3_key}"
    return public_url

def generate_public_urls_for_objects(s3_keys, bucket=None):
    """Generate public URLs for multiple S3 objects"""
    if bucket is None:
        bucket = S3_CONFIG['bucket']
        
    urls = {}
    for key in tqdm(s3_keys, desc="Generating public URLs"):
        url = generate_public_url(key, bucket)
        urls[key] = url
    
    return urls

def save_urls_to_file(urls, output_file="public_urls.txt"):
    """Save generated URLs to a file"""
    with open(output_file, 'w') as f:
        f.write(f"# Generated on: {datetime.datetime.now()}\n")
        
        for key, url in urls.get('urls', {}).items():
            f.write(f"File: {key}\n")
            f.write(f"URL: {url}\n\n")
    
    print(f"Saved {len(urls.get('urls', {}))} URLs to {output_file}")
    return output_file

def list_bucket_objects(prefix=None, bucket=None):
    """List objects in the S3 bucket with optional prefix filter"""
    if bucket is None:
        bucket = S3_CONFIG['bucket']
    
    s3_client = initialize_s3_client()
    
    try:
        if prefix:
            response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        else:
            response = s3_client.list_objects_v2(Bucket=bucket)
            
        if 'Contents' in response:
            return [obj['Key'] for obj in response['Contents']]
        else:
            return []
            
    except ClientError as e:
        print(f"Error listing objects in bucket {bucket}: {e}")
        return []

def check_bucket_public_access(bucket=None):
    """Check if the bucket has public access enabled and proper bucket policy"""
    if bucket is None:
        bucket = S3_CONFIG['bucket']
    
    s3_client = initialize_s3_client()
    s3_resource = boto3.resource(
        's3',
        aws_access_key_id=S3_CONFIG['aws_access_key_id'],
        aws_secret_access_key=S3_CONFIG['aws_secret_access_key'],
        region_name=S3_CONFIG['region_name']
    )
    
    print(f"🔍 Checking public access settings for bucket: {bucket}")
    
    # Check Block Public Access settings
    try:
        public_access_block = s3_client.get_public_access_block(Bucket=bucket)
        block_settings = public_access_block['PublicAccessBlockConfiguration']
        
        if (block_settings['BlockPublicAcls'] or 
            block_settings['IgnorePublicAcls'] or 
            block_settings['BlockPublicPolicy'] or 
            block_settings['RestrictPublicBuckets']):
            
            print("❌ Bucket has public access blocks enabled:")
            print(f"   - Block public ACLs: {block_settings['BlockPublicAcls']}")
            print(f"   - Ignore public ACLs: {block_settings['IgnorePublicAcls']}")
            print(f"   - Block public policy: {block_settings['BlockPublicPolicy']}")
            print(f"   - Restrict public buckets: {block_settings['RestrictPublicBuckets']}")
            print("\n✏️ Solution: Disable these settings in AWS console:")
            print("   S3 > Buckets > your-bucket > Permissions > Block public access > Edit")
            return False
        else:
            print("✅ Bucket public access blocks are disabled correctly")
    except Exception as e:
        # Some older buckets might not have public access block settings
        print("ℹ️ Could not retrieve public access block settings (this might be normal for older buckets)")
    
    # Check bucket policy
    try:
        bucket_policy = s3_client.get_bucket_policy(Bucket=bucket)
        print("✅ Bucket has a policy defined")
        # Check if policy contains public read permissions
        policy_str = bucket_policy['Policy']
        if '"Effect":"Allow"' in policy_str and '"Principal":"*"' in policy_str and '"Action":"s3:GetObject"' in policy_str:
            print("✅ Bucket policy appears to allow public read access")
        else:
            print("⚠️ Bucket has a policy, but it may not allow public read access")
            print("✏️ Consider adding a public read policy using --make-public")
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchBucketPolicy':
            print("❌ Bucket has no policy defined")
            print("✏️ Solution: Add a public read policy using --make-public")
        else:
            print(f"❌ Error checking bucket policy: {e}")
    
    # Test public access by attempting to download a file
    print("\n🔍 Testing public access to a sample object...")
    try:
        # List objects to find a sample
        response = s3_client.list_objects_v2(Bucket=bucket, MaxKeys=1)
        if 'Contents' in response and len(response['Contents']) > 0:
            sample_key = response['Contents'][0]['Key']
            public_url = generate_public_url(sample_key, bucket)
            
            import requests
            print(f"   Testing access to: {public_url}")
            r = requests.head(public_url, timeout=10)
            
            if r.status_code == 200:
                print(f"✅ Public access works! Successfully accessed {sample_key}")
                return True
            else:
                print(f"❌ Public access test failed with HTTP status: {r.status_code}")
                return False
        else:
            print("ℹ️ Could not test public access - no objects in bucket")
    except Exception as e:
        print(f"❌ Error testing public access: {str(e)}")
    
    return False

def add_public_read_bucket_policy(bucket=None):
    """Add a bucket policy that allows public read access"""
    if bucket is None:
        bucket = S3_CONFIG['bucket']
    
    s3_client = initialize_s3_client()
    
    # Create a bucket policy that allows GetObject
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "PublicReadGetObject",
                "Effect": "Allow",
                "Principal": "*",
                "Action": "s3:GetObject",
                "Resource": f"arn:aws:s3:::{bucket}/*"
            }
        ]
    }
    
    import json
    policy_str = json.dumps(policy)
    
    try:
        print(f"📝 Adding public read policy to bucket {bucket}...")
        s3_client.put_bucket_policy(Bucket=bucket, Policy=policy_str)
        print("✅ Successfully added public read policy")
        return True
    except Exception as e:
        print(f"❌ Error adding bucket policy: {str(e)}")
        return False

def make_objects_public(prefix=None, bucket=None):
    """Make objects public by adding public-read ACL to each object"""
    if bucket is None:
        bucket = S3_CONFIG['bucket']
    
    s3_client = initialize_s3_client()
    s3_resource = boto3.resource(
        's3',
        aws_access_key_id=S3_CONFIG['aws_access_key_id'],
        aws_secret_access_key=S3_CONFIG['aws_secret_access_key'],
        region_name=S3_CONFIG['region_name']
    )
    
    # Get list of objects
    object_keys = list_bucket_objects(prefix, bucket)
    if not object_keys:
        print(f"No objects found with prefix: {prefix}")
        return False
    
    print(f"Found {len(object_keys)} objects. Making them publicly readable...")
    failures = 0
    
    for key in tqdm(object_keys, desc="Setting public-read ACL"):
        try:
            s3_resource.ObjectAcl(bucket, key).put(ACL='public-read')
        except Exception as e:
            print(f"❌ Error setting ACL for {key}: {str(e)}")
            failures += 1
    
    if failures:
        print(f"⚠️ Completed with {failures} failures out of {len(object_keys)} objects")
    else:
        print(f"✅ Successfully made {len(object_keys)} objects public")
    
    return failures == 0

def test_object_public_access(s3_key, bucket=None):
    """Test if an object is publicly accessible via HTTP"""
    if bucket is None:
        bucket = S3_CONFIG['bucket']
    
    public_url = generate_public_url(s3_key, bucket)
    
    print(f"🔍 Testing public access to: {public_url}")
    try:
        import requests
        r = requests.head(public_url, timeout=10)
        
        if r.status_code == 200:
            print(f"✅ Success! Object is publicly accessible (HTTP {r.status_code})")
            return True
        else:
            print(f"❌ Failed with HTTP status: {r.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing access: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Upload files to S3 and generate public URLs')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Upload command
    upload_parser = subparsers.add_parser('upload', help='Upload files to S3')
    upload_parser.add_argument('directory', type=str, help='Local directory path to upload')
    upload_parser.add_argument('--prefix', type=str, default=S3_CONFIG['default_prefix'],
                       help=f'S3 prefix (folder path) for uploaded files (default: {S3_CONFIG["default_prefix"]})')
    upload_parser.add_argument('--no-subdirs', action='store_true',
                       help='Do not include subdirectories')
    upload_parser.add_argument('--workers', type=int, default=10,
                       help='Maximum number of parallel upload workers')
    
    # URL command
    url_parser = subparsers.add_parser('url', help='Generate public URLs for S3 objects')
    url_parser.add_argument('--prefix', type=str, default=S3_CONFIG['default_prefix'],
                      help=f'S3 prefix to filter objects (default: {S3_CONFIG["default_prefix"]})')
    url_parser.add_argument('--file', type=str, help='Specific S3 object key to generate URL for')
    url_parser.add_argument('--output', type=str, default="public_urls.txt",
                      help='Output file to save URLs (default: public_urls.txt)')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download a file from S3')
    download_parser.add_argument('--file', type=str, required=True,
                        help='S3 object key to download')
    download_parser.add_argument('--output', type=str, default=None,
                        help='Local path to save the file (default: use filename from S3 key)')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify S3 credentials')
    
    # Public access management command
    public_parser = subparsers.add_parser('public', help='Manage public access to S3 objects')
    public_parser.add_argument('--check', action='store_true',
                      help='Check bucket public access settings')
    public_parser.add_argument('--add-policy', action='store_true',
                      help='Add a public read bucket policy')
    public_parser.add_argument('--make-public', type=str, nargs='?', const='',
                      help='Make objects with prefix publicly readable (defaults to all objects)')
    public_parser.add_argument('--test', type=str,
                      help='Test if a specific object is publicly accessible')
    
    args = parser.parse_args()
    
    # Default to help if no command specified
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Verify command - check credentials
    if args.command == 'verify':
        print(f"Verifying AWS credentials for bucket: {S3_CONFIG['bucket']} in region: {S3_CONFIG['region_name']}")
        verified = verify_s3_credentials()
        if not verified:
            sys.exit(1)
        sys.exit(0)
        
    # Other commands - verify credentials first
    if not verify_s3_credentials():
        print("\n❌ Cannot proceed due to credential verification failure")
        sys.exit(1)
    
    # Handle upload command
    if args.command == 'upload':
        print(f"Starting upload of '{args.directory}' to s3://{S3_CONFIG['bucket']}/{args.prefix}")
        
        upload_directory_to_s3(
            directory_path=args.directory,
            s3_prefix=args.prefix,
            include_subdirs=not args.no_subdirs,
            max_workers=args.workers
        )
        
        print("Upload process complete!")
    
    # Handle URL command
    elif args.command == 'url':
        if args.file:
            # Generate URL for a single file
            print(f"Generating public URL for s3://{S3_CONFIG['bucket']}/{args.file}")
            url = generate_public_url(args.file)
            
            print(f"\nPublic URL:")
            print(url)
            
            # Save to file
            urls = {
                'urls': {args.file: url}
            }
            save_urls_to_file(urls, args.output)
            
            # Print curl command for easy download
            print("\nTo download with curl:")
            filename = os.path.basename(args.file)
            print(f"curl -o {filename} '{url}'")
        else:
            # Generate URLs for all files with prefix
            print(f"Listing objects in s3://{S3_CONFIG['bucket']}/{args.prefix}")
            s3_keys = list_bucket_objects(args.prefix)
            
            if not s3_keys:
                print(f"No objects found with prefix: {args.prefix}")
                sys.exit(0)
                
            print(f"Found {len(s3_keys)} objects. Generating public URLs...")
            urls = generate_public_urls_for_objects(s3_keys)
            
            # Save to file
            url_data = {
                'urls': urls
            }
            output_file = save_urls_to_file(url_data, args.output)
            print(f"URLs have been saved to {output_file}")
            print("\nNOTE: These URLs will only work if your objects are publicly accessible.")
            print("Make sure your bucket or objects have public read permissions configured.")
    
    # Handle download command
    elif args.command == 'download':
        download_file(args.file, args.output)
    
    # Handle public command
    elif args.command == 'public':
        if args.check:
            check_bucket_public_access()
        
        if args.add_policy:
            add_public_read_bucket_policy()
            print("\nAfter adding policy, checking bucket settings again:")
            check_bucket_public_access()
        
        if args.make_public is not None:
            prefix = args.make_public
            print(f"Making objects with prefix '{prefix}' publicly readable...")
            make_objects_public(prefix)
        
        if args.test:
            test_object_public_access(args.test)
            
        # If no specific action was chosen, run a complete check
        if not any([args.check, args.add_policy, args.make_public is not None, args.test]):
            check_bucket_public_access()

if __name__ == "__main__":
    main()
