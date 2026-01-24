// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <utility>

namespace hipdnn_ep {

/// Status code for Status class.
enum class StatusCode {
  kSuccess,
  kFailure,
};

/// Lightweight status class for internal error handling.
/// Stores success/failure state and an error message on failure.
class Status {
 public:
  /// Create a success status.
  static Status Success() { return Status(); }

  /// Create a failure status with a message.
  static Status Failure(std::string message) {
    return Status(StatusCode::kFailure, std::move(message));
  }

  /// Check if the status indicates success.
  bool ok() const { return code_ == StatusCode::kSuccess; }

  /// Check if the status indicates failure.
  bool failed() const { return code_ == StatusCode::kFailure; }

  /// Get the status code.
  StatusCode code() const { return code_; }

  /// Get the error message (empty if success).
  const std::string& message() const { return message_; }

 private:
  /// Default constructor creates success status.
  Status() : code_(StatusCode::kSuccess) {}

  /// Private constructor for failure status.
  Status(StatusCode code, std::string message)
      : code_(code), message_(std::move(message)) {}

  StatusCode code_;
  std::string message_;
};

}  // namespace hipdnn_ep
